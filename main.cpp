#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ============================================================
// ARTICLE-BASED 3D CANNY + MOTION-TOLERANT 3D TRACKING + TAYLOR
// ============================================================
//
// Scientific basis:
//   F. Pirillo, H. Shao, S. Petro, G. Moroni (2024),
//   "A Surface Determination Technique for Dimensional and
//    Geometrical Analysis in Industrial X-ray Computed Tomography"
//
// The paper's core pipeline is preserved:
//   1) Gaussian smoothing of the XCT volume;
//   2) gradient computation in x/y/z;
//   3) non-maximum suppression along the 3D gradient direction;
//   4) Canny-style edge tracking by hysteresis;
//   5) subvoxel refinement along the normalized gradient using
//      the third-order Taylor model
//          t* = -D_g^2 f / D_g^3 f.
//
// Engineering additions in this version (NOT claimed as part of
// Pirillo et al. itself):
//   A) local 3D gradient-orientation coherence computed from a
//      structure-tensor-like 3x3x3 neighbourhood;
//   B) grey-value contrast measured across the candidate normal;
//   C) support from neighbouring Z slices using predicted surface
//      motion and normal agreement;
//   D) a composite surface score used for hysteresis thresholds;
//   E) normal-compatible 3D hysteresis to avoid following unrelated
//      texture/ray artefacts;
//   F) persistent internal-surface seeds: hole walls may seed hysteresis
//      from strong Z-persistence even when they are weaker than the outer
//      boundary in the global score distribution;
//   G) evidence-guided short-gap bridging over EXISTING 3D-NMS points;
//   H) a very broad intensity ROI is used as a computational/spatial
//      support mask. It is filled and strongly dilated, so it does not
//      define the boundary; it only prevents spending most of the runtime
//      on remote reconstruction streaks;
//   I) topology-preserving subvoxel fallback when Taylor is numerically
//      unstable at an otherwise accepted surface point;
//   J) V4 motion-tolerant Z tracking: inner walls are matched recursively
//      from slice to slice, allowing the contour to drift in X/Y;
//   K) V4 motion-aware hysteresis: cross-slice edge tracking follows the
//      predicted 3D surface motion rather than only the 26-neighbourhood.
//
// IMPORTANT:
// This is deliberately a TEST-WINDOW implementation. First tune and
// inspect a local stack (e.g. 25 slices). Do not run the full 1903-slice
// dataset until the diagnostics look correct. A full-volume version is
// better implemented block-wise afterwards.
// ============================================================

// ============================================================
// PATHS / TEST WINDOW
// ============================================================

static const std::string INPUT_FOLDER =
    "D:/Studies/Real_work/BladeCT/"
    "20251010_Blade_16bit_75.40um_1804x910x1903";

static const std::string OUTPUT_FOLDER =
    "D:/Studies/Real_work/BladeCT/Results/article_surface_v4_final";

constexpr int FIRST_SLICE  = 994;
constexpr int CENTER_SLICE = 1006;
constexpr int LAST_SLICE   = 1018;

// ============================================================
// PHYSICAL VOXEL SIZE, micrometres
// ============================================================

constexpr float SPACING_X = 75.4f;
constexpr float SPACING_Y = 75.4f;
constexpr float SPACING_Z = 75.4f;

// ============================================================
// GAUSSIAN SMOOTHING
// ============================================================
// Moderate smoothing: enough to stabilise derivatives but not enough
// to erase the thin blade boundary and small holes.
// ============================================================

constexpr double SIGMA_XY = 1.15;
constexpr double SIGMA_Z  = 1.00;
constexpr int Z_GAUSS_RADIUS = 3;

// ============================================================
// 3D NMS
// ============================================================
// The paper/Canny logic: keep local maxima of |grad f| along the
// local gradient direction. A single local step is intentionally used;
// several large NMS distances can suppress a real broad XCT edge.
// ============================================================

constexpr float NMS_STEP = 1.0f;
constexpr float NMS_PLATEAU_FACTOR = 0.975f;

// We only sample a subset of NMS candidates for percentile statistics;
// every candidate itself is still kept and processed.
constexpr int STAT_SAMPLE_XY_STRIDE = 2;
constexpr int STAT_SAMPLE_Z_STRIDE  = 1;

// ============================================================
// 3D QUALITY METRICS - OUR ADDITIONS
// ============================================================

// Local structure-tensor-like coherence neighbourhood.
constexpr int COHERENCE_RADIUS = 1; // 3 x 3 x 3
constexpr float COHERENCE_SIGMA = 1.0f;
constexpr int COHERENCE_POWER_ITERATIONS = 5;

// Measure the material/air (or material/material) jump across the
// candidate normal at two distances and average them.
constexpr float CONTRAST_STEP_NEAR = 0.90f;
constexpr float CONTRAST_STEP_FAR  = 1.80f;
constexpr float CONTRAST_NEAR_WEIGHT = 0.65f;
constexpr float CONTRAST_FAR_WEIGHT  = 0.35f;

// Support from neighbouring slices. The surface position is predicted
// from the local 3D normal, then a small XY neighbourhood is searched.
constexpr int Z_SUPPORT_RADIUS = 4;

// V4: support is tracked slice-by-slice instead of comparing every
// neighbour slice only with the original seed position. This is important
// for inner hole walls that drift by a few pixels between adjacent slices.
constexpr int Z_SUPPORT_SEARCH_RADIUS = 2;
constexpr int Z_SUPPORT_RECOVERY_SEARCH_RADIUS = 3;
constexpr int Z_SUPPORT_MAX_CONSECUTIVE_MISSES = 1;

// Local orientation is more important than orientation relative to the
// original seed, because the wall can curve while it moves through Z.
constexpr float Z_SUPPORT_MIN_NORMAL_COSINE = 0.38f;
constexpr float Z_SUPPORT_MIN_SEED_NORMAL_COSINE = 0.28f;
constexpr float Z_SUPPORT_MIN_MAG_RATIO = 0.09f;
constexpr float Z_SUPPORT_LOCAL_NORMAL_WEIGHT = 0.70f;
constexpr float Z_SUPPORT_SEED_NORMAL_WEIGHT  = 0.30f;
constexpr float Z_SUPPORT_MIN_MATCH_SCORE = 0.10f;

// Protect prediction from an unstable near-Z normal.
constexpr float Z_SUPPORT_MAX_SHIFT_PER_SLICE = 5.0f;

// Robust normalisation of gradient and contrast before composing score.
constexpr float GRADIENT_SCALE_PERCENTILE = 96.0f;
constexpr float CONTRAST_SCALE_PERCENTILE = 94.0f;

// Composite surface score:
//   score = gNorm * (
//       base + wCoh*coherence + wContrast*contrastNorm + wZ*zSupportNorm)
// We keep a non-zero base so a genuine but locally incomplete contour
// can still survive; coherence/contrast/Z merely increase confidence.
constexpr float SCORE_BASE       = 0.35f;
constexpr float SCORE_W_COH      = 0.30f;
constexpr float SCORE_W_CONTRAST = 0.20f;
constexpr float SCORE_W_Z        = 0.15f;

// ============================================================
// 3D HYSTERESIS ON SURFACE SCORE
// ============================================================
// The old run used weak=28th and strong=88th percentile of raw gradient
// maxima and therefore accepted ~8.17M of ~11.74M NMS voxels. Here the
// thresholds are applied to the richer surface score and the weak level
// is intentionally much less permissive.
// ============================================================

constexpr float STRONG_SCORE_PERCENTILE = 92.0f;
constexpr float WEAK_SCORE_PERCENTILE   = 60.0f;
constexpr float WEAK_TO_STRONG_RATIO    = 0.48f;

// Rescue slightly sub-threshold fragments when their 3D evidence is
// very good. They still must connect to a strong edge through hysteresis.
constexpr float RESCUE_TO_WEAK_RATIO = 0.82f;
constexpr int RESCUE_MIN_Z_SUPPORT = 3;
constexpr float RESCUE_MIN_COHERENCE = 0.72f;
constexpr float RESCUE_MIN_CONTRAST_NORM = 0.18f;

// During hysteresis, a weak neighbour should locally belong to the same
// surface. Curved surfaces are allowed, hence this is deliberately mild.
constexpr float HYSTERESIS_MIN_NORMAL_COSINE = 0.45f;
constexpr int HYSTERESIS_Z_SUPPORT_OVERRIDE = 4;
constexpr float HYSTERESIS_COHERENCE_OVERRIDE = 0.84f;

// V4: cross-slice hysteresis follows the predicted surface motion instead
// of being limited to the ordinary 26-neighbourhood. Only the best
// candidate in each adjacent slice is attached to the track.
constexpr int HYSTERESIS_TRACK_SEARCH_RADIUS = 1;
constexpr float HYSTERESIS_TRACK_MIN_NORMAL_COSINE = 0.30f;
constexpr int HYSTERESIS_TRACK_MIN_Z_SUPPORT = 2;

// ============================================================
// PERSISTENT INTERNAL-SURFACE SEEDS
// ============================================================
// A global strong percentile naturally favours the very bright outer
// metal/air boundary. Internal hole walls can be physically real and
// highly persistent in Z while never reaching that global percentile.
// We therefore allow such NMS points to become independent hysteresis
// seeds, but ONLY inside the undilated filled object silhouette.
constexpr int INTERNAL_CORE_ERODE_RADIUS = 1;
constexpr float PERSISTENT_SEED_TO_WEAK_RATIO = 0.70f;
constexpr int PERSISTENT_SEED_MIN_Z_SUPPORT = 4;      // out of max 8
constexpr float PERSISTENT_SEED_MIN_COHERENCE = 0.54f;
constexpr float PERSISTENT_SEED_MIN_CONTRAST_NORM = 0.08f;

// ============================================================
// SHORT CONTOUR-GAP BRIDGING ON EXISTING NMS CANDIDATES
// ============================================================
// This is not a blind morphological closing. A point may be restored
// only when:
//   * it is already a 3D-NMS candidate;
//   * it lies inside the object core;
//   * it has non-trivial 3D evidence;
//   * accepted contour points exist on BOTH tangent sides.
// Thus the pass reconnects short missing pieces without inventing an
// arbitrary straight line through the volume.
constexpr float BRIDGE_SCORE_TO_WEAK_RATIO = 0.36f;
constexpr int BRIDGE_MIN_Z_SUPPORT = 2;
constexpr float BRIDGE_MIN_COHERENCE = 0.36f;
constexpr float BRIDGE_MIN_CONTRAST_NORM = 0.05f;
constexpr int BRIDGE_TANGENT_RADIUS = 10;
constexpr int BRIDGE_SIDE_WINDOW_RADIUS = 2;
constexpr int BRIDGE_PASSES = 6;
constexpr float BRIDGE_MIN_NORMAL_COSINE = 0.18f;

// ============================================================
// BROAD POST-DETECTION OBJECT ROI
// ============================================================
// This is NOT the boundary detector. It is only a generous spatial crop
// after Canny/hysteresis, used to delete remote streaks/ray artefacts.
// ============================================================

constexpr int ROI_CLOSE_RADIUS = 4;
constexpr int ROI_DILATE_RADIUS = 20;
constexpr int ROI_MIN_COMPONENT_AREA = 200;
constexpr float ROI_BRIGHT_PERCENTILE = 72.0f;

// ============================================================
// TAYLOR SUBVOXEL REFINEMENT
// ============================================================
// The paper computes derivatives of f and uses
//      t* = -D_g^2 f / D_g^3 f.
//
// To keep memory reasonable, this implementation does NOT allocate 19
// derivative volumes. Instead, for every accepted edge voxel we evaluate
// the required first/second/third partial derivatives locally from the
// smoothed 3D field, then contract them with the local unit gradient.
// This follows the same mathematical equations while avoiding several
// gigabytes of additional derivative arrays.
// ============================================================

constexpr float TAYLOR_MAX_ABS_DISPLACEMENT_VOXELS = 1.35f;
constexpr float TAYLOR_MIN_ABS_D3 = 1e-13f;
constexpr bool TAYLOR_REQUIRE_LOCAL_MAXIMUM = true;

// If the third-order fit becomes numerically invalid, do not create a
// topological hole in an otherwise accepted Canny surface. First try a
// local parabolic maximum of |grad f| along the same 3D normal. If even
// that is not stable, keep the accepted voxel centre (t = 0).
constexpr bool ENABLE_SUBVOXEL_FALLBACK = true;
constexpr float PARABOLIC_MAX_ABS_OFFSET = 0.75f;

// ============================================================
// OUTPUT OPTIONS
// ============================================================

constexpr bool SAVE_ACCEPTED_MASK_FOR_ALL_TEST_SLICES = true;

// ============================================================
// DATA TYPES
// ============================================================

struct GradientVoxel
{
    float gx = 0.0f;
    float gy = 0.0f;
    float gz = 0.0f;
    float mag = 0.0f;
};

struct GradientVolume
{
    int width = 0;
    int height = 0;
    int depth = 0;
    std::vector<GradientVoxel> data;

    inline size_t index(int x, int y, int z) const
    {
        return
            (static_cast<size_t>(z) * static_cast<size_t>(height) +
             static_cast<size_t>(y)) *
                static_cast<size_t>(width) +
            static_cast<size_t>(x);
    }

    inline GradientVoxel& at(int x, int y, int z)
    {
        return data[index(x, y, z)];
    }

    inline const GradientVoxel& at(int x, int y, int z) const
    {
        return data[index(x, y, z)];
    }
};

struct SurfaceMetricsVolume
{
    int width = 0;
    int height = 0;
    int depth = 0;

    // meaningful only where NMS != 0
    std::vector<float> coherence;
    std::vector<float> contrast;
    std::vector<float> contrastNorm;
    std::vector<float> score;
    std::vector<uint8_t> zSupport;

    float gradientScale = 1.0f;
    float contrastScale = 1.0f;

    inline size_t index(int x, int y, int z) const
    {
        return
            (static_cast<size_t>(z) * static_cast<size_t>(height) +
             static_cast<size_t>(y)) *
                static_cast<size_t>(width) +
            static_cast<size_t>(x);
    }
};

struct SubvoxelPoint
{
    int sourceX = 0;
    int sourceY = 0;
    int sourceZ = 0;

    float tPhysicalUm = 0.0f;

    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    float physicalX = 0.0f;
    float physicalY = 0.0f;
    float physicalZ = 0.0f;

    float gradientMagnitude = 0.0f;
    float coherence = 0.0f;
    float contrast = 0.0f;
    int zSupport = 0;
    float surfaceScore = 0.0f;

    // 1 = third-order Taylor (article)
    // 2 = parabolic |grad f| fallback
    // 3 = accepted voxel centre fallback
    int refinementType = 0;
};

// ============================================================
// BASIC HELPERS
// ============================================================

static float clamp01(float v)
{
    return std::max(0.0f, std::min(1.0f, v));
}

static float percentile(std::vector<float> values, float p)
{
    if (values.empty())
        return 0.0f;

    p = std::clamp(p, 0.0f, 100.0f);

    const size_t index =
        static_cast<size_t>(
            (p / 100.0f) *
            static_cast<float>(values.size() - 1)
            );

    std::nth_element(
        values.begin(),
        values.begin() + static_cast<std::ptrdiff_t>(index),
        values.end()
        );

    return values[index];
}

static cv::Vec3f normalFromGradient(const GradientVoxel& g)
{
    if (g.mag <= 1e-12f)
        return cv::Vec3f(0.0f, 0.0f, 0.0f);

    return cv::Vec3f(
        g.gx / g.mag,
        g.gy / g.mag,
        g.gz / g.mag
        );
}

// Physical-space unit normal -> voxel-index direction.
static cv::Vec3f normalVoxelDirection(const cv::Vec3f& normal)
{
    const float reference =
        std::min({SPACING_X, SPACING_Y, SPACING_Z});

    cv::Vec3f d(
        normal[0] * reference / SPACING_X,
        normal[1] * reference / SPACING_Y,
        normal[2] * reference / SPACING_Z
        );

    const float len =
        std::sqrt(
            d[0] * d[0] +
            d[1] * d[1] +
            d[2] * d[2]
            );

    if (len > 1e-12f)
        d /= len;

    return d;
}

static float absNormalCosine(
    const GradientVoxel& a,
    const GradientVoxel& b)
{
    if (a.mag <= 1e-12f || b.mag <= 1e-12f)
        return 0.0f;

    const float dot =
        (a.gx * b.gx + a.gy * b.gy + a.gz * b.gz) /
        (a.mag * b.mag);

    return std::abs(std::clamp(dot, -1.0f, 1.0f));
}

// ============================================================
// FILE COLLECTION / IO
// ============================================================

static std::vector<fs::path> collectTiffFiles(const std::string& folder)
{
    std::vector<fs::path> files;

    if (!fs::exists(folder))
    {
        std::cerr << "Input folder does not exist:\n"
                  << folder << std::endl;
        return files;
    }

    for (const auto& entry : fs::directory_iterator(folder))
    {
        if (!entry.is_regular_file())
            continue;

        std::string ext = entry.path().extension().string();
        std::transform(
            ext.begin(), ext.end(), ext.begin(),
            [](unsigned char c)
            {
                return static_cast<char>(std::tolower(c));
            }
            );

        if (ext == ".tif" || ext == ".tiff")
            files.push_back(entry.path());
    }

    std::sort(files.begin(), files.end());
    return files;
}

static bool loadFloatSlice(const fs::path& file, cv::Mat& out)
{
    cv::Mat src = cv::imread(file.string(), cv::IMREAD_UNCHANGED);

    if (src.empty())
    {
        std::cerr << "Cannot read: " << file << std::endl;
        return false;
    }

    if (src.channels() != 1)
    {
        std::cerr << "Expected single-channel TIFF: " << file << std::endl;
        return false;
    }

    if (src.type() == CV_16UC1)
    {
        src.convertTo(out, CV_32F, 1.0 / 65535.0);
    }
    else if (src.type() == CV_8UC1)
    {
        src.convertTo(out, CV_32F, 1.0 / 255.0);
    }
    else
    {
        double minValue = 0.0;
        double maxValue = 0.0;
        cv::minMaxLoc(src, &minValue, &maxValue);

        if (maxValue <= minValue)
        {
            std::cerr << "Constant or invalid image: " << file << std::endl;
            return false;
        }

        src.convertTo(
            out,
            CV_32F,
            1.0 / (maxValue - minValue),
            -minValue / (maxValue - minValue)
            );
    }

    return true;
}

static cv::Mat makeDisplayImage(const cv::Mat& src)
{
    cv::Mat f;
    src.convertTo(f, CV_32F);

    std::vector<float> samples;
    samples.reserve(
        static_cast<size_t>((f.rows + 1) / 2) *
        static_cast<size_t>((f.cols + 1) / 2)
        );

    for (int y = 0; y < f.rows; y += 2)
    {
        const float* row = f.ptr<float>(y);
        for (int x = 0; x < f.cols; x += 2)
            samples.push_back(row[x]);
    }

    const float lo = percentile(samples, 1.0f);
    const float hi = percentile(samples, 99.7f);
    const float range = std::max(hi - lo, 1e-6f);

    cv::Mat display(f.size(), CV_8U, cv::Scalar(0));

    for (int y = 0; y < f.rows; ++y)
    {
        const float* s = f.ptr<float>(y);
        uchar* d = display.ptr<uchar>(y);

        for (int x = 0; x < f.cols; ++x)
        {
            const float t = clamp01((s[x] - lo) / range);
            d[x] = cv::saturate_cast<uchar>(255.0f * t);
        }
    }

    return display;
}

// ============================================================
// LOAD LOCAL 3D VOLUME + XY GAUSSIAN
// ============================================================

static bool loadVolumeXYSmoothed(
    const std::vector<fs::path>& files,
    std::vector<float>& volume,
    int& width,
    int& height,
    int& depth)
{
    depth = LAST_SLICE - FIRST_SLICE + 1;

    cv::Mat first;
    if (!loadFloatSlice(files[FIRST_SLICE], first))
        return false;

    width = first.cols;
    height = first.rows;

    const size_t sliceSize =
        static_cast<size_t>(width) *
        static_cast<size_t>(height);

    volume.resize(sliceSize * static_cast<size_t>(depth));

    std::cout << "Loading and XY-Gaussian smoothing "
              << depth << " slices...\n";

    for (int z = 0; z < depth; ++z)
    {
        const int globalZ = FIRST_SLICE + z;

        cv::Mat src;
        if (!loadFloatSlice(files[globalZ], src))
            return false;

        if (src.cols != width || src.rows != height)
        {
            std::cerr << "Image dimensions changed at slice "
                      << globalZ << std::endl;
            return false;
        }

        cv::Mat blurred;
        cv::GaussianBlur(
            src,
            blurred,
            cv::Size(0, 0),
            SIGMA_XY,
            SIGMA_XY,
            cv::BORDER_REPLICATE
            );

        float* dst =
            volume.data() +
            static_cast<size_t>(z) * sliceSize;

        std::memcpy(
            dst,
            blurred.ptr<float>(),
            sliceSize * sizeof(float)
            );

        std::cout << "\rLoaded " << globalZ
                  << " / " << LAST_SLICE
                  << std::flush;
    }

    std::cout << std::endl;
    return true;
}

// ============================================================
// GAUSSIAN ALONG Z
// ============================================================

static std::vector<float> createGaussianKernelZ()
{
    std::vector<float> kernel(2 * Z_GAUSS_RADIUS + 1, 0.0f);

    float sum = 0.0f;

    for (int k = -Z_GAUSS_RADIUS; k <= Z_GAUSS_RADIUS; ++k)
    {
        const float x = static_cast<float>(k);
        const float v =
            std::exp(
                -(x * x) /
                static_cast<float>(2.0 * SIGMA_Z * SIGMA_Z)
                );

        kernel[k + Z_GAUSS_RADIUS] = v;
        sum += v;
    }

    for (float& v : kernel)
        v /= sum;

    return kernel;
}

static void gaussianBlurZInPlace(
    std::vector<float>& volume,
    int width,
    int height,
    int depth)
{
    std::cout << "Gaussian smoothing along Z..." << std::endl;

    const size_t sliceSize =
        static_cast<size_t>(width) *
        static_cast<size_t>(height);

    const std::vector<float> kernel = createGaussianKernelZ();
    std::vector<float> output(volume.size(), 0.0f);

    cv::parallel_for_(
        cv::Range(0, depth),
        [&](const cv::Range& range)
        {
            for (int z = range.start; z < range.end; ++z)
            {
                float* dst =
                    output.data() +
                    static_cast<size_t>(z) * sliceSize;

                for (int y = 0; y < height; ++y)
                {
                    for (int x = 0; x < width; ++x)
                    {
                        float sum = 0.0f;

                        for (int k = -Z_GAUSS_RADIUS;
                             k <= Z_GAUSS_RADIUS;
                             ++k)
                        {
                            const int zz =
                                std::clamp(z + k, 0, depth - 1);

                            const size_t idx =
                                static_cast<size_t>(zz) * sliceSize +
                                static_cast<size_t>(y) * width +
                                static_cast<size_t>(x);

                            sum +=
                                volume[idx] *
                                kernel[k + Z_GAUSS_RADIUS];
                        }

                        dst[static_cast<size_t>(y) * width + x] = sum;
                    }
                }
            }
        }
        );

    volume.swap(output);
}

// ============================================================
// VOLUME SAMPLING
// ============================================================

static float sampleVolumeTrilinear(
    const std::vector<float>& volume,
    int width,
    int height,
    int depth,
    float x,
    float y,
    float z)
{
    x = std::clamp(x, 0.0f, static_cast<float>(width - 1));
    y = std::clamp(y, 0.0f, static_cast<float>(height - 1));
    z = std::clamp(z, 0.0f, static_cast<float>(depth - 1));

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int z0 = static_cast<int>(std::floor(z));

    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);
    const int z1 = std::min(z0 + 1, depth - 1);

    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const float tz = z - static_cast<float>(z0);

    const size_t sliceSize =
        static_cast<size_t>(width) *
        static_cast<size_t>(height);

    auto v = [&](int xx, int yy, int zz) -> float
    {
        return volume[
            static_cast<size_t>(zz) * sliceSize +
            static_cast<size_t>(yy) * width +
            static_cast<size_t>(xx)
        ];
    };

    const float c000 = v(x0, y0, z0);
    const float c100 = v(x1, y0, z0);
    const float c010 = v(x0, y1, z0);
    const float c110 = v(x1, y1, z0);
    const float c001 = v(x0, y0, z1);
    const float c101 = v(x1, y0, z1);
    const float c011 = v(x0, y1, z1);
    const float c111 = v(x1, y1, z1);

    const float c00 = c000 * (1.0f - tx) + c100 * tx;
    const float c10 = c010 * (1.0f - tx) + c110 * tx;
    const float c01 = c001 * (1.0f - tx) + c101 * tx;
    const float c11 = c011 * (1.0f - tx) + c111 * tx;

    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;

    return c0 * (1.0f - tz) + c1 * tz;
}

static float voxelValue(
    const std::vector<float>& volume,
    int width,
    int height,
    int depth,
    int x,
    int y,
    int z)
{
    x = std::clamp(x, 0, width - 1);
    y = std::clamp(y, 0, height - 1);
    z = std::clamp(z, 0, depth - 1);

    const size_t sliceSize =
        static_cast<size_t>(width) *
        static_cast<size_t>(height);

    return volume[
        static_cast<size_t>(z) * sliceSize +
        static_cast<size_t>(y) * width +
        static_cast<size_t>(x)
    ];
}

// ============================================================
// 3D GRADIENT
// ============================================================
// Article: Gaussian smoothing, then partial derivatives in x/y/z with
// Sobel operators. OpenCV's Sobel is 2D, therefore x/y are true Sobel
// derivatives on each already Z-smoothed slice, while z is a centred
// derivative of the same smoothed 3D field. All three derivatives are
// expressed per physical micrometre, so anisotropic spacing is supported.
// ============================================================

static GradientVolume computeGradient3D(
    const std::vector<float>& volume,
    int width,
    int height,
    int depth)
{
    std::cout << "Computing 3D X/Y/Z gradient..." << std::endl;

    GradientVolume result;
    result.width = width;
    result.height = height;
    result.depth = depth;

    const size_t sliceSize =
        static_cast<size_t>(width) *
        static_cast<size_t>(height);

    result.data.resize(sliceSize * static_cast<size_t>(depth));

    cv::parallel_for_(
        cv::Range(0, depth),
        [&](const cv::Range& range)
        {
            for (int z = range.start; z < range.end; ++z)
            {
                cv::Mat slice(
                    height,
                    width,
                    CV_32F,
                    const_cast<float*>(
                        volume.data() +
                        static_cast<size_t>(z) * sliceSize
                        )
                    );

                cv::Mat gx;
                cv::Mat gy;

                cv::Sobel(
                    slice,
                    gx,
                    CV_32F,
                    1,
                    0,
                    3,
                    1.0 / (8.0 * SPACING_X),
                    0.0,
                    cv::BORDER_REPLICATE
                    );

                cv::Sobel(
                    slice,
                    gy,
                    CV_32F,
                    0,
                    1,
                    3,
                    1.0 / (8.0 * SPACING_Y),
                    0.0,
                    cv::BORDER_REPLICATE
                    );

                for (int y = 0; y < height; ++y)
                {
                    const float* gxRow = gx.ptr<float>(y);
                    const float* gyRow = gy.ptr<float>(y);

                    for (int x = 0; x < width; ++x)
                    {
                        GradientVoxel& g = result.at(x, y, z);
                        g.gx = gxRow[x];
                        g.gy = gyRow[x];
                    }
                }
            }
        }
        );

    cv::parallel_for_(
        cv::Range(0, depth),
        [&](const cv::Range& range)
        {
            for (int z = range.start; z < range.end; ++z)
            {
                const int zm = std::max(0, z - 1);
                const int zp = std::min(depth - 1, z + 1);

                const float dzDistance =
                    static_cast<float>(zp - zm) * SPACING_Z;

                for (int y = 0; y < height; ++y)
                {
                    for (int x = 0; x < width; ++x)
                    {
                        const size_t idxM =
                            static_cast<size_t>(zm) * sliceSize +
                            static_cast<size_t>(y) * width + x;

                        const size_t idxP =
                            static_cast<size_t>(zp) * sliceSize +
                            static_cast<size_t>(y) * width + x;

                        GradientVoxel& g = result.at(x, y, z);

                        g.gz =
                            (dzDistance > 0.0f)
                                ? (volume[idxP] - volume[idxM]) / dzDistance
                                : 0.0f;

                        g.mag =
                            std::sqrt(
                                g.gx * g.gx +
                                g.gy * g.gy +
                                g.gz * g.gz
                                );
                    }
                }
            }
        }
        );

    return result;
}

// ============================================================
// TRILINEAR GRADIENT-MAGNITUDE SAMPLE
// ============================================================

static float sampleMagnitude(
    const GradientVolume& gradient,
    float x,
    float y,
    float z)
{
    if (x < 0.0f || y < 0.0f || z < 0.0f ||
        x >= static_cast<float>(gradient.width - 1) ||
        y >= static_cast<float>(gradient.height - 1) ||
        z >= static_cast<float>(gradient.depth - 1))
    {
        return 0.0f;
    }

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int z0 = static_cast<int>(std::floor(z));

    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;

    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const float tz = z - static_cast<float>(z0);

    auto m = [&](int xx, int yy, int zz) -> float
    {
        return gradient.at(xx, yy, zz).mag;
    };

    const float c000 = m(x0, y0, z0);
    const float c100 = m(x1, y0, z0);
    const float c010 = m(x0, y1, z0);
    const float c110 = m(x1, y1, z0);
    const float c001 = m(x0, y0, z1);
    const float c101 = m(x1, y0, z1);
    const float c011 = m(x0, y1, z1);
    const float c111 = m(x1, y1, z1);

    const float c00 = c000 * (1.0f - tx) + c100 * tx;
    const float c10 = c010 * (1.0f - tx) + c110 * tx;
    const float c01 = c001 * (1.0f - tx) + c101 * tx;
    const float c11 = c011 * (1.0f - tx) + c111 * tx;

    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;

    return c0 * (1.0f - tz) + c1 * tz;
}

// ============================================================
// TRUE 3D NMS
// ============================================================

static std::vector<uint8_t> computeNMS3D(
    const GradientVolume& gradient,
    std::vector<float>& gradientSamples)
{
    std::cout << "Computing 3D non-maximum suppression..." << std::endl;

    const size_t total = gradient.data.size();
    std::vector<uint8_t> nms(total, 0);

    const size_t sliceSize =
        static_cast<size_t>(gradient.width) *
        static_cast<size_t>(gradient.height);

    cv::parallel_for_(
        cv::Range(1, gradient.depth - 1),
        [&](const cv::Range& range)
        {
            for (int z = range.start; z < range.end; ++z)
            {
                for (int y = 1; y < gradient.height - 1; ++y)
                {
                    for (int x = 1; x < gradient.width - 1; ++x)
                    {
                        const GradientVoxel& g = gradient.at(x, y, z);

                        if (g.mag <= 1e-12f)
                            continue;

                        const cv::Vec3f normal = normalFromGradient(g);
                        const cv::Vec3f d = normalVoxelDirection(normal);

                        const float plus =
                            sampleMagnitude(
                                gradient,
                                static_cast<float>(x) + d[0] * NMS_STEP,
                                static_cast<float>(y) + d[1] * NMS_STEP,
                                static_cast<float>(z) + d[2] * NMS_STEP
                                );

                        const float minus =
                            sampleMagnitude(
                                gradient,
                                static_cast<float>(x) - d[0] * NMS_STEP,
                                static_cast<float>(y) - d[1] * NMS_STEP,
                                static_cast<float>(z) - d[2] * NMS_STEP
                                );

                        const bool maximum =
                            g.mag >= plus * NMS_PLATEAU_FACTOR &&
                            g.mag >= minus * NMS_PLATEAU_FACTOR;

                        if (!maximum)
                            continue;

                        const size_t idx =
                            static_cast<size_t>(z) * sliceSize +
                            static_cast<size_t>(y) * gradient.width +
                            static_cast<size_t>(x);

                        nms[idx] = 1;
                    }
                }
            }
        }
        );

    gradientSamples.clear();

    for (int z = 1;
         z < gradient.depth - 1;
         z += STAT_SAMPLE_Z_STRIDE)
    {
        for (int y = 1;
             y < gradient.height - 1;
             y += STAT_SAMPLE_XY_STRIDE)
        {
            for (int x = 1;
                 x < gradient.width - 1;
                 x += STAT_SAMPLE_XY_STRIDE)
            {
                const size_t idx =
                    static_cast<size_t>(z) * sliceSize +
                    static_cast<size_t>(y) * gradient.width +
                    static_cast<size_t>(x);

                if (nms[idx] != 0)
                    gradientSamples.push_back(gradient.data[idx].mag);
            }
        }
    }

    std::cout << "NMS statistical samples: "
              << gradientSamples.size() << std::endl;

    return nms;
}

// ============================================================
// 3D STRUCTURE-TENSOR-LIKE COHERENCE
// ============================================================
// We accumulate J = sum(w * grad*grad^T) in a 3x3x3 neighbourhood.
// The ratio lambda_max / trace(J) is close to 1 when local gradients
// mostly point in one normal direction (surface-like), and ~1/3 for
// isotropic/noisy orientation distributions. We remap [1/3,1] -> [0,1].
// ============================================================

static float computeLocalGradientCoherence(
    const GradientVolume& gradient,
    int x,
    int y,
    int z)
{
    double jxx = 0.0;
    double jxy = 0.0;
    double jxz = 0.0;
    double jyy = 0.0;
    double jyz = 0.0;
    double jzz = 0.0;

    const double inv2Sigma2 =
        1.0 / (2.0 * COHERENCE_SIGMA * COHERENCE_SIGMA);

    for (int dz = -COHERENCE_RADIUS;
         dz <= COHERENCE_RADIUS;
         ++dz)
    {
        const int zz = z + dz;
        if (zz < 0 || zz >= gradient.depth)
            continue;

        for (int dy = -COHERENCE_RADIUS;
             dy <= COHERENCE_RADIUS;
             ++dy)
        {
            const int yy = y + dy;
            if (yy < 0 || yy >= gradient.height)
                continue;

            for (int dx = -COHERENCE_RADIUS;
                 dx <= COHERENCE_RADIUS;
                 ++dx)
            {
                const int xx = x + dx;
                if (xx < 0 || xx >= gradient.width)
                    continue;

                const GradientVoxel& g = gradient.at(xx, yy, zz);

                if (g.mag <= 1e-12f)
                    continue;

                const double r2 =
                    static_cast<double>(dx * dx + dy * dy + dz * dz);

                const double w = std::exp(-r2 * inv2Sigma2);

                const double gx = g.gx;
                const double gy = g.gy;
                const double gz = g.gz;

                jxx += w * gx * gx;
                jxy += w * gx * gy;
                jxz += w * gx * gz;
                jyy += w * gy * gy;
                jyz += w * gy * gz;
                jzz += w * gz * gz;
            }
        }
    }

    const double trace = jxx + jyy + jzz;
    if (trace <= 1e-24)
        return 0.0f;

    cv::Vec3d v(
        gradient.at(x, y, z).gx,
        gradient.at(x, y, z).gy,
        gradient.at(x, y, z).gz
        );

    double len = cv::norm(v);
    if (len <= 1e-18)
        v = cv::Vec3d(1.0, 0.0, 0.0);
    else
        v /= len;

    for (int iter = 0; iter < COHERENCE_POWER_ITERATIONS; ++iter)
    {
        cv::Vec3d next(
            jxx * v[0] + jxy * v[1] + jxz * v[2],
            jxy * v[0] + jyy * v[1] + jyz * v[2],
            jxz * v[0] + jyz * v[1] + jzz * v[2]
            );

        const double n = cv::norm(next);
        if (n <= 1e-24)
            break;

        v = next / n;
    }

    const cv::Vec3d jv(
        jxx * v[0] + jxy * v[1] + jxz * v[2],
        jxy * v[0] + jyy * v[1] + jyz * v[2],
        jxz * v[0] + jyz * v[1] + jzz * v[2]
        );

    const double lambdaMax = v.dot(jv);
    const double dominance = std::clamp(lambdaMax / trace, 0.0, 1.0);

    const double mapped =
        (dominance - 1.0 / 3.0) /
        (2.0 / 3.0);

    return clamp01(static_cast<float>(mapped));
}

// ============================================================
// CONTRAST ACROSS LOCAL NORMAL
// ============================================================

static float computeNormalContrast(
    const std::vector<float>& volume,
    int width,
    int height,
    int depth,
    const GradientVolume& gradient,
    int x,
    int y,
    int z)
{
    const GradientVoxel& g = gradient.at(x, y, z);
    if (g.mag <= 1e-12f)
        return 0.0f;

    const cv::Vec3f normal = normalFromGradient(g);
    const cv::Vec3f d = normalVoxelDirection(normal);

    auto pairContrast = [&](float step) -> float
    {
        const float plus =
            sampleVolumeTrilinear(
                volume,
                width,
                height,
                depth,
                static_cast<float>(x) + d[0] * step,
                static_cast<float>(y) + d[1] * step,
                static_cast<float>(z) + d[2] * step
                );

        const float minus =
            sampleVolumeTrilinear(
                volume,
                width,
                height,
                depth,
                static_cast<float>(x) - d[0] * step,
                static_cast<float>(y) - d[1] * step,
                static_cast<float>(z) - d[2] * step
                );

        return std::abs(plus - minus);
    };

    return
        CONTRAST_NEAR_WEIGHT * pairContrast(CONTRAST_STEP_NEAR) +
        CONTRAST_FAR_WEIGHT  * pairContrast(CONTRAST_STEP_FAR);
}

// ============================================================
// PREDICT SURFACE SHIFT BETWEEN Z SLICES
// ============================================================
// For a local level surface F(x,y,z)=const:
//     Fx dx + Fy dy + Fz dz = 0.
// We choose the minimum-norm XY displacement along the XY projection
// of the normal. Returned shift is in voxel coordinates.
// ============================================================

static cv::Point2f predictSurfaceShift(
    const cv::Vec3f& normal,
    int deltaZ)
{
    const float nx = normal[0];
    const float ny = normal[1];
    const float nz = normal[2];

    const float denominator = nx * nx + ny * ny;

    if (denominator < 1e-6f)
        return cv::Point2f(0.0f, 0.0f);

    const float dzPhysical =
        static_cast<float>(deltaZ) * SPACING_Z;

    const float dxPhysical =
        -nz * dzPhysical * nx / denominator;

    const float dyPhysical =
        -nz * dzPhysical * ny / denominator;

    cv::Point2f shift(
        dxPhysical / SPACING_X,
        dyPhysical / SPACING_Y
        );

    const float length =
        std::sqrt(shift.x * shift.x + shift.y * shift.y);

    const float maxLength =
        Z_SUPPORT_MAX_SHIFT_PER_SLICE *
        static_cast<float>(std::max(1, std::abs(deltaZ)));

    if (length > maxLength && length > 1e-6f)
    {
        const float scale = maxLength / length;
        shift.x *= scale;
        shift.y *= scale;
    }

    return shift;
}

// ============================================================
// Z SUPPORT FOR ONE NMS CANDIDATE
// ============================================================

struct ZTrackCandidate
{
    bool found = false;
    int x = 0;
    int y = 0;
    float match = 0.0f;
};

static ZTrackCandidate findBestZTrackCandidate(
    const GradientVolume& gradient,
    const std::vector<uint8_t>& nms,
    const GradientVoxel& seed,
    const GradientVoxel& previous,
    float expectedX,
    float expectedY,
    int z,
    int searchRadius)
{
    ZTrackCandidate best;

    const size_t sliceSize =
        static_cast<size_t>(gradient.width) *
        static_cast<size_t>(gradient.height);

    const int cx = cvRound(expectedX);
    const int cy = cvRound(expectedY);

    for (int oy = -searchRadius; oy <= searchRadius; ++oy)
    {
        const int yy = cy + oy;
        if (yy < 1 || yy >= gradient.height - 1)
            continue;

        for (int ox = -searchRadius; ox <= searchRadius; ++ox)
        {
            const int xx = cx + ox;
            if (xx < 1 || xx >= gradient.width - 1)
                continue;

            const size_t idx =
                static_cast<size_t>(z) * sliceSize +
                static_cast<size_t>(yy) * gradient.width +
                static_cast<size_t>(xx);

            if (nms[idx] == 0)
                continue;

            const GradientVoxel& candidate = gradient.data[idx];
            if (candidate.mag <= 1e-12f)
                continue;

            const float localCos =
                absNormalCosine(previous, candidate);

            const float seedCos =
                absNormalCosine(seed, candidate);

            // Allow a curved / drifting wall when either the local track
            // orientation remains compatible or there is still reasonable
            // agreement with the original seed.
            if (localCos < Z_SUPPORT_MIN_NORMAL_COSINE &&
                seedCos < Z_SUPPORT_MIN_SEED_NORMAL_COSINE)
            {
                continue;
            }

            const float magRatio =
                std::min(previous.mag, candidate.mag) /
                std::max(previous.mag, candidate.mag);

            if (magRatio < Z_SUPPORT_MIN_MAG_RATIO)
                continue;

            const float orientation =
                Z_SUPPORT_LOCAL_NORMAL_WEIGHT * localCos +
                Z_SUPPORT_SEED_NORMAL_WEIGHT  * seedCos;

            const float dist2 =
                static_cast<float>(ox * ox + oy * oy);

            const float distPenalty =
                1.0f / (1.0f + 0.12f * dist2);

            // sqrt(magRatio) is deliberately milder than multiplying by
            // magRatio directly. Inner walls often change contrast from one
            // slice to the next because of partial-volume effects.
            const float magnitudeTerm =
                0.30f + 0.70f * std::sqrt(std::max(0.0f, magRatio));

            const float match =
                orientation * magnitudeTerm * distPenalty;

            if (match >= Z_SUPPORT_MIN_MATCH_SCORE &&
                (!best.found || match > best.match))
            {
                best.found = true;
                best.x = xx;
                best.y = yy;
                best.match = match;
            }
        }
    }

    return best;
}

static uint8_t computeCandidateZSupport(
    const GradientVolume& gradient,
    const std::vector<uint8_t>& nms,
    int x,
    int y,
    int z)
{
    const GradientVoxel& seed = gradient.at(x, y, z);
    if (seed.mag <= 1e-12f)
        return 0;

    int hits = 0;

    // Track independently towards -Z and +Z. Crucially, every successful
    // match becomes the predictor for the next slice. Therefore a hole wall
    // may drift / curve through the stack instead of having to remain close
    // to one line extrapolated from the centre slice.
    for (int direction : {-1, +1})
    {
        int lastX = x;
        int lastY = y;
        int lastZ = z;

        GradientVoxel previous = seed;
        int consecutiveMisses = 0;

        for (int step = 1; step <= Z_SUPPORT_RADIUS; ++step)
        {
            const int zz = z + direction * step;
            if (zz < 0 || zz >= gradient.depth)
                break;

            const int deltaFromLast = zz - lastZ;

            const cv::Vec3f previousNormal =
                normalFromGradient(previous);

            const cv::Point2f predicted =
                predictSurfaceShift(previousNormal, deltaFromLast);

            const float expectedX =
                static_cast<float>(lastX) + predicted.x;

            const float expectedY =
                static_cast<float>(lastY) + predicted.y;

            const int searchRadius =
                (consecutiveMisses == 0)
                    ? Z_SUPPORT_SEARCH_RADIUS
                    : Z_SUPPORT_RECOVERY_SEARCH_RADIUS;

            const ZTrackCandidate match =
                findBestZTrackCandidate(
                    gradient,
                    nms,
                    seed,
                    previous,
                    expectedX,
                    expectedY,
                    zz,
                    searchRadius
                    );

            if (match.found)
            {
                ++hits;
                consecutiveMisses = 0;

                lastX = match.x;
                lastY = match.y;
                lastZ = zz;
                previous = gradient.at(match.x, match.y, zz);
            }
            else
            {
                ++consecutiveMisses;

                if (consecutiveMisses >
                    Z_SUPPORT_MAX_CONSECUTIVE_MISSES)
                {
                    break;
                }

                // Do not move the anchor on a miss. At the next step
                // deltaFromLast becomes 2, so the same 3D normal predicts
                // the location two slices away and the wider recovery
                // window gets one chance to reacquire the surface.
            }
        }
    }

    return static_cast<uint8_t>(hits);
}

// ============================================================
// COMPUTE QUALITY METRICS FOR NMS CANDIDATES
// ============================================================

static SurfaceMetricsVolume computeSurfaceMetrics(
    const std::vector<float>& volume,
    int width,
    int height,
    int depth,
    const GradientVolume& gradient,
    const std::vector<uint8_t>& nms,
    const std::vector<uint8_t>& broadROI3D,
    const std::vector<float>& gradientSamples,
    std::vector<float>& scoreSamples)
{
    std::cout << "Computing 3D surface quality metrics..." << std::endl;

    SurfaceMetricsVolume metrics;
    metrics.width = width;
    metrics.height = height;
    metrics.depth = depth;

    const size_t total = gradient.data.size();
    const size_t sliceSize =
        static_cast<size_t>(width) *
        static_cast<size_t>(height);

    metrics.coherence.assign(total, 0.0f);
    metrics.contrast.assign(total, 0.0f);
    metrics.contrastNorm.assign(total, 0.0f);
    metrics.score.assign(total, 0.0f);
    metrics.zSupport.assign(total, 0);

    metrics.gradientScale =
        std::max(
            percentile(gradientSamples, GRADIENT_SCALE_PERCENTILE),
            1e-12f
            );

    // Pass 1: expensive geometric features only on NMS candidates.
    cv::parallel_for_(
        cv::Range(1, depth - 1),
        [&](const cv::Range& range)
        {
            for (int z = range.start; z < range.end; ++z)
            {
                for (int y = 1; y < height - 1; ++y)
                {
                    for (int x = 1; x < width - 1; ++x)
                    {
                        const size_t idx =
                            static_cast<size_t>(z) * sliceSize +
                            static_cast<size_t>(y) * width +
                            static_cast<size_t>(x);

                        if (nms[idx] == 0 || broadROI3D[idx] == 0)
                            continue;

                        metrics.coherence[idx] =
                            computeLocalGradientCoherence(
                                gradient, x, y, z
                                );

                        metrics.contrast[idx] =
                            computeNormalContrast(
                                volume,
                                width,
                                height,
                                depth,
                                gradient,
                                x,
                                y,
                                z
                                );

                        metrics.zSupport[idx] =
                            computeCandidateZSupport(
                                gradient,
                                nms,
                                x,
                                y,
                                z
                                );
                    }
                }
            }
        }
        );

    std::vector<float> contrastSamples;

    for (int z = 1; z < depth - 1; z += STAT_SAMPLE_Z_STRIDE)
    {
        for (int y = 1; y < height - 1; y += STAT_SAMPLE_XY_STRIDE)
        {
            for (int x = 1; x < width - 1; x += STAT_SAMPLE_XY_STRIDE)
            {
                const size_t idx =
                    static_cast<size_t>(z) * sliceSize +
                    static_cast<size_t>(y) * width +
                    static_cast<size_t>(x);

                if (nms[idx] != 0 && broadROI3D[idx] != 0)
                    contrastSamples.push_back(metrics.contrast[idx]);
            }
        }
    }

    metrics.contrastScale =
        std::max(
            percentile(contrastSamples, CONTRAST_SCALE_PERCENTILE),
            1e-8f
            );

    std::cout
        << "  gradient scale P" << GRADIENT_SCALE_PERCENTILE
        << " = " << metrics.gradientScale << '\n'
        << "  contrast scale P" << CONTRAST_SCALE_PERCENTILE
        << " = " << metrics.contrastScale << std::endl;

    // Pass 2: compose score.
    cv::parallel_for_(
        cv::Range(1, depth - 1),
        [&](const cv::Range& range)
        {
            for (int z = range.start; z < range.end; ++z)
            {
                for (int y = 1; y < height - 1; ++y)
                {
                    for (int x = 1; x < width - 1; ++x)
                    {
                        const size_t idx =
                            static_cast<size_t>(z) * sliceSize +
                            static_cast<size_t>(y) * width +
                            static_cast<size_t>(x);

                        if (nms[idx] == 0 || broadROI3D[idx] == 0)
                            continue;

                        const float gNorm =
                            clamp01(
                                gradient.data[idx].mag /
                                metrics.gradientScale
                                );

                        const float contrastNorm =
                            clamp01(
                                metrics.contrast[idx] /
                                metrics.contrastScale
                                );

                        metrics.contrastNorm[idx] = contrastNorm;

                        const float zNorm =
                            clamp01(
                                static_cast<float>(metrics.zSupport[idx]) /
                                static_cast<float>(2 * Z_SUPPORT_RADIUS)
                                );

                        const float quality =
                            SCORE_BASE +
                            SCORE_W_COH * metrics.coherence[idx] +
                            SCORE_W_CONTRAST * contrastNorm +
                            SCORE_W_Z * zNorm;

                        metrics.score[idx] = gNorm * quality;
                    }
                }
            }
        }
        );

    scoreSamples.clear();

    for (int z = 1; z < depth - 1; z += STAT_SAMPLE_Z_STRIDE)
    {
        for (int y = 1; y < height - 1; y += STAT_SAMPLE_XY_STRIDE)
        {
            for (int x = 1; x < width - 1; x += STAT_SAMPLE_XY_STRIDE)
            {
                const size_t idx =
                    static_cast<size_t>(z) * sliceSize +
                    static_cast<size_t>(y) * width +
                    static_cast<size_t>(x);

                if (nms[idx] != 0 && broadROI3D[idx] != 0)
                    scoreSamples.push_back(metrics.score[idx]);
            }
        }
    }

    std::cout << "Surface-score statistical samples: "
              << scoreSamples.size() << std::endl;

    return metrics;
}

// ============================================================
// 3D CANNY-LIKE HYSTERESIS ON SURFACE SCORE
// ============================================================
// edgeType:
//   0 = rejected/non-NMS
//   1 = weak
//   2 = strong seed
//   3 = 3D-rescue candidate
//
// accepted is generated by growing only from strong seeds.
// ============================================================

static void hysteresis3D(
    const GradientVolume& gradient,
    const std::vector<uint8_t>& nms,
    const SurfaceMetricsVolume& metrics,
    const std::vector<uint8_t>& coreROI,
    float weakThreshold,
    float strongThreshold,
    std::vector<uint8_t>& edgeType,
    std::vector<uint8_t>& accepted)
{
    std::cout << "Running quality-aware 3D hysteresis..." << std::endl;

    const size_t total = gradient.data.size();
    edgeType.assign(total, 0);
    accepted.assign(total, 0);

    const float rescueThreshold =
        weakThreshold * RESCUE_TO_WEAK_RATIO;

    const float persistentSeedThreshold =
        weakThreshold * PERSISTENT_SEED_TO_WEAK_RATIO;

    std::vector<uint32_t> queue;
    queue.reserve(total / 30);

    long long strongCount = 0;
    long long persistentSeedCount = 0;
    long long weakCount = 0;
    long long rescueCount = 0;

    for (size_t i = 0; i < total; ++i)
    {
        if (nms[i] == 0)
            continue;

        const float score = metrics.score[i];

        // Global strong Canny seeds.
        if (score >= strongThreshold)
        {
            edgeType[i] = 2;
            accepted[i] = 1;
            queue.push_back(static_cast<uint32_t>(i));
            ++strongCount;
        }
        // Internal boundaries (especially hole walls) can be weaker than
        // the outer metal/air boundary, so a global percentile may never
        // make them strong. If the same candidate is persistent through
        // many neighbouring slices and has coherent 3D gradient evidence,
        // promote it to an independent seed. Restricting this to coreROI
        // prevents this mechanism from seeding exterior ray artefacts.
        else if (
            coreROI[i] != 0 &&
            score >= persistentSeedThreshold &&
            metrics.zSupport[i] >= PERSISTENT_SEED_MIN_Z_SUPPORT &&
            metrics.coherence[i] >= PERSISTENT_SEED_MIN_COHERENCE &&
            metrics.contrastNorm[i] >= PERSISTENT_SEED_MIN_CONTRAST_NORM)
        {
            edgeType[i] = 4;
            accepted[i] = 1;
            queue.push_back(static_cast<uint32_t>(i));
            ++persistentSeedCount;
        }
        else if (score >= weakThreshold)
        {
            edgeType[i] = 1;
            ++weakCount;
        }
        else if (
            score >= rescueThreshold &&
            metrics.zSupport[i] >= RESCUE_MIN_Z_SUPPORT &&
            metrics.coherence[i] >= RESCUE_MIN_COHERENCE &&
            metrics.contrastNorm[i] >= RESCUE_MIN_CONTRAST_NORM)
        {
            edgeType[i] = 3;
            ++rescueCount;
        }
    }

    std::cout
        << "  strong seeds        = " << strongCount << '\n'
        << "  persistent 3D seeds = " << persistentSeedCount << '\n'
        << "  weak candidates     = " << weakCount << '\n'
        << "  rescue candidates   = " << rescueCount << std::endl;

    const int width = gradient.width;
    const int height = gradient.height;
    const int depth = gradient.depth;

    const size_t sliceSize =
        static_cast<size_t>(width) *
        static_cast<size_t>(height);

    size_t head = 0;

    auto canAcceptCandidate =
        [&](size_t currentIdx,
            size_t candidateIdx,
            float minimumCosine,
            bool requireTrackSupport) -> bool
    {
        if (accepted[candidateIdx] != 0)
            return false;

        if (edgeType[candidateIdx] == 0)
            return false;

        const float cosine =
            absNormalCosine(
                gradient.data[currentIdx],
                gradient.data[candidateIdx]
                );

        const bool compatibleNormal =
            cosine >= minimumCosine;

        const bool strong3DOverride =
            metrics.zSupport[candidateIdx] >=
                HYSTERESIS_Z_SUPPORT_OVERRIDE ||
            metrics.coherence[candidateIdx] >=
                HYSTERESIS_COHERENCE_OVERRIDE;

        if (requireTrackSupport)
        {
            const bool tracked =
                metrics.zSupport[candidateIdx] >=
                HYSTERESIS_TRACK_MIN_Z_SUPPORT;

            if (!(compatibleNormal && tracked) &&
                !strong3DOverride)
            {
                return false;
            }
        }
        else
        {
            if (!compatibleNormal && !strong3DOverride)
                return false;
        }

        return true;
    };

    while (head < queue.size())
    {
        const uint32_t linear = queue[head++];

        const int z =
            static_cast<int>(
                static_cast<size_t>(linear) / sliceSize
                );

        const size_t insideSlice =
            static_cast<size_t>(linear) -
            static_cast<size_t>(z) * sliceSize;

        const int y =
            static_cast<int>(insideSlice / static_cast<size_t>(width));

        const int x =
            static_cast<int>(
                insideSlice -
                static_cast<size_t>(y) * static_cast<size_t>(width)
                );

        const GradientVoxel& current = gradient.data[linear];

        // ----------------------------------------------------
        // A) ordinary 8-neighbour tracking INSIDE the same slice
        // ----------------------------------------------------
        for (int dy = -1; dy <= 1; ++dy)
        {
            const int yy = y + dy;
            if (yy < 0 || yy >= height)
                continue;

            for (int dx = -1; dx <= 1; ++dx)
            {
                if (dx == 0 && dy == 0)
                    continue;

                const int xx = x + dx;
                if (xx < 0 || xx >= width)
                    continue;

                const size_t nidx =
                    static_cast<size_t>(z) * sliceSize +
                    static_cast<size_t>(yy) * width +
                    static_cast<size_t>(xx);

                if (!canAcceptCandidate(
                        static_cast<size_t>(linear),
                        nidx,
                        HYSTERESIS_MIN_NORMAL_COSINE,
                        false))
                {
                    continue;
                }

                accepted[nidx] = 1;
                queue.push_back(static_cast<uint32_t>(nidx));
            }
        }

        // ----------------------------------------------------
        // B) motion-aware tracking to z-1 and z+1
        // ----------------------------------------------------
        // The old 26-neighbourhood could move at most one XY pixel while
        // changing a slice. Real inner holes in this scan drift by more
        // than that. We predict the intersection shift from the local 3D
        // normal and attach only the BEST compatible candidate around the
        // predicted point on each adjacent slice.
        const cv::Vec3f currentNormal =
            normalFromGradient(current);

        for (int dz : {-1, +1})
        {
            const int zz = z + dz;
            if (zz < 0 || zz >= depth)
                continue;

            const cv::Point2f predicted =
                predictSurfaceShift(currentNormal, dz);

            const int cx =
                cvRound(static_cast<float>(x) + predicted.x);

            const int cy =
                cvRound(static_cast<float>(y) + predicted.y);

            size_t bestIdx = std::numeric_limits<size_t>::max();
            float bestMatch = -1.0f;

            for (int oy = -HYSTERESIS_TRACK_SEARCH_RADIUS;
                 oy <= HYSTERESIS_TRACK_SEARCH_RADIUS;
                 ++oy)
            {
                const int yy = cy + oy;
                if (yy < 0 || yy >= height)
                    continue;

                for (int ox = -HYSTERESIS_TRACK_SEARCH_RADIUS;
                     ox <= HYSTERESIS_TRACK_SEARCH_RADIUS;
                     ++ox)
                {
                    const int xx = cx + ox;
                    if (xx < 0 || xx >= width)
                        continue;

                    const size_t nidx =
                        static_cast<size_t>(zz) * sliceSize +
                        static_cast<size_t>(yy) * width +
                        static_cast<size_t>(xx);

                    if (!canAcceptCandidate(
                            static_cast<size_t>(linear),
                            nidx,
                            HYSTERESIS_TRACK_MIN_NORMAL_COSINE,
                            true))
                    {
                        continue;
                    }

                    const float cosine =
                        absNormalCosine(
                            current,
                            gradient.data[nidx]
                            );

                    const float dist2 =
                        static_cast<float>(ox * ox + oy * oy);

                    const float distPenalty =
                        1.0f / (1.0f + 0.18f * dist2);

                    const float supportBonus =
                        1.0f +
                        0.06f *
                            static_cast<float>(metrics.zSupport[nidx]);

                    const float match =
                        cosine * distPenalty * supportBonus;

                    if (match > bestMatch)
                    {
                        bestMatch = match;
                        bestIdx = nidx;
                    }
                }
            }

            if (bestIdx != std::numeric_limits<size_t>::max())
            {
                accepted[bestIdx] = 1;
                queue.push_back(static_cast<uint32_t>(bestIdx));
            }
        }
    }

    std::cout << "Accepted by quality-aware 3D hysteresis: "
              << queue.size() << std::endl;
}

// ============================================================
// EVIDENCE-GUIDED SHORT GAP BRIDGING
// ============================================================
// Standard hysteresis is intentionally conservative and can leave a
// short break in an otherwise excellent hole contour. We do NOT use a
// binary morphological closing here. Instead, we restore only existing
// 3D-NMS candidates which are:
//   - inside the object core,
//   - persistent in neighbouring slices,
//   - supported by local gradient quality,
//   - geometrically located between already accepted contour points
//     on both tangent sides in the same slice.
//
// This makes the operation topology-aware without moving the measured
// boundary away from the gradient ridge.
// ============================================================

static bool hasAcceptedOnTangentSide(
    const GradientVolume& gradient,
    const std::vector<uint8_t>& accepted,
    int x,
    int y,
    int z,
    float tx,
    float ty,
    int sign)
{
    const size_t sliceSize =
        static_cast<size_t>(gradient.width) *
        static_cast<size_t>(gradient.height);

    const GradientVoxel& seed = gradient.at(x, y, z);

    for (int step = 2; step <= BRIDGE_TANGENT_RADIUS; ++step)
    {
        const float px =
            static_cast<float>(x) +
            static_cast<float>(sign * step) * tx;

        const float py =
            static_cast<float>(y) +
            static_cast<float>(sign * step) * ty;

        const int cx = cvRound(px);
        const int cy = cvRound(py);

        for (int oy = -BRIDGE_SIDE_WINDOW_RADIUS;
             oy <= BRIDGE_SIDE_WINDOW_RADIUS;
             ++oy)
        {
            const int yy = cy + oy;
            if (yy < 0 || yy >= gradient.height)
                continue;

            for (int ox = -BRIDGE_SIDE_WINDOW_RADIUS;
                 ox <= BRIDGE_SIDE_WINDOW_RADIUS;
                 ++ox)
            {
                const int xx = cx + ox;
                if (xx < 0 || xx >= gradient.width)
                    continue;

                const float projection =
                    (static_cast<float>(xx - x) * tx +
                     static_cast<float>(yy - y) * ty) *
                    static_cast<float>(sign);

                // Ensure that the found point really lies on the requested
                // side, not in the search window around the origin.
                if (projection < 1.25f)
                    continue;

                const size_t idx =
                    static_cast<size_t>(z) * sliceSize +
                    static_cast<size_t>(yy) * gradient.width +
                    static_cast<size_t>(xx);

                if (accepted[idx] == 0)
                    continue;

                const float cosine =
                    absNormalCosine(seed, gradient.data[idx]);

                if (cosine >= BRIDGE_MIN_NORMAL_COSINE)
                    return true;
            }
        }
    }

    return false;
}

static long long bridgeShortContourGaps3D(
    const GradientVolume& gradient,
    const std::vector<uint8_t>& nms,
    const SurfaceMetricsVolume& metrics,
    const std::vector<uint8_t>& coreROI,
    float weakThreshold,
    std::vector<uint8_t>& edgeType,
    std::vector<uint8_t>& accepted)
{
    std::cout << "Bridging short internal contour gaps..." << std::endl;

    const int width = gradient.width;
    const int height = gradient.height;
    const int depth = gradient.depth;

    const size_t sliceSize =
        static_cast<size_t>(width) *
        static_cast<size_t>(height);

    const float bridgeScoreThreshold =
        weakThreshold * BRIDGE_SCORE_TO_WEAK_RATIO;

    long long totalAdded = 0;

    for (int pass = 0; pass < BRIDGE_PASSES; ++pass)
    {
        std::vector<uint32_t> add;
        add.reserve(4096);

        for (int z = 1; z < depth - 1; ++z)
        {
            for (int y = 2; y < height - 2; ++y)
            {
                for (int x = 2; x < width - 2; ++x)
                {
                    const size_t idx =
                        static_cast<size_t>(z) * sliceSize +
                        static_cast<size_t>(y) * width +
                        static_cast<size_t>(x);

                    if (accepted[idx] != 0 ||
                        nms[idx] == 0 ||
                        coreROI[idx] == 0)
                    {
                        continue;
                    }

                    if (metrics.score[idx] < bridgeScoreThreshold)
                        continue;

                    if (metrics.zSupport[idx] < BRIDGE_MIN_Z_SUPPORT)
                        continue;

                    if (metrics.coherence[idx] < BRIDGE_MIN_COHERENCE)
                        continue;

                    if (metrics.contrastNorm[idx] <
                        BRIDGE_MIN_CONTRAST_NORM)
                    {
                        continue;
                    }

                    const GradientVoxel& g = gradient.data[idx];
                    if (g.mag <= 1e-12f)
                        continue;

                    const cv::Vec3f n = normalFromGradient(g);

                    const float xy =
                        std::sqrt(n[0] * n[0] + n[1] * n[1]);

                    // A nearly Z-normal surface does not form a reliable
                    // 2D contour tangent in this slice.
                    if (xy < 0.10f)
                        continue;

                    const float tx = -n[1] / xy;
                    const float ty =  n[0] / xy;

                    const bool forward =
                        hasAcceptedOnTangentSide(
                            gradient,
                            accepted,
                            x, y, z,
                            tx, ty,
                            +1
                            );

                    if (!forward)
                        continue;

                    const bool backward =
                        hasAcceptedOnTangentSide(
                            gradient,
                            accepted,
                            x, y, z,
                            tx, ty,
                            -1
                            );

                    if (!backward)
                        continue;

                    add.push_back(static_cast<uint32_t>(idx));
                }
            }
        }

        if (add.empty())
        {
            std::cout
                << "  bridge pass " << (pass + 1)
                << ": no new points" << std::endl;
            break;
        }

        for (const uint32_t idx : add)
        {
            if (accepted[idx] == 0)
            {
                accepted[idx] = 1;
                edgeType[idx] = 5;
                ++totalAdded;
            }
        }

        std::cout
            << "  bridge pass " << (pass + 1)
            << ": +" << add.size()
            << " candidates" << std::endl;
    }

    std::cout
        << "Total evidence-guided bridge points: "
        << totalAdded << std::endl;

    return totalAdded;
}

// ============================================================
// BROAD OBJECT ROI FOR POST-FILTERING ONLY
// ============================================================

static cv::Mat fillBinaryHoles(const cv::Mat& binary)
{
    cv::Mat flood = binary.clone();

    cv::copyMakeBorder(
        flood,
        flood,
        1, 1, 1, 1,
        cv::BORDER_CONSTANT,
        cv::Scalar(0)
        );

    cv::floodFill(flood, cv::Point(0, 0), cv::Scalar(255));

    cv::Mat floodCrop =
        flood(cv::Rect(1, 1, binary.cols, binary.rows)).clone();

    cv::Mat holes;
    cv::bitwise_not(floodCrop, holes);

    cv::Mat filled;
    cv::bitwise_or(binary, holes, filled);

    return filled;
}

static cv::Mat buildBroadObjectROI(
    const cv::Mat& centerFloat,
    cv::Mat* coreOut = nullptr)
{
    std::vector<float> samples;
    samples.reserve(
        static_cast<size_t>((centerFloat.rows + 1) / 2) *
        static_cast<size_t>((centerFloat.cols + 1) / 2)
        );

    for (int y = 0; y < centerFloat.rows; y += 2)
    {
        const float* row = centerFloat.ptr<float>(y);
        for (int x = 0; x < centerFloat.cols; x += 2)
            samples.push_back(row[x]);
    }

    const float low = percentile(samples, 1.0f);
    const float high = percentile(samples, 99.5f);
    const float range = std::max(high - low, 1e-6f);
    const float pThreshold = percentile(samples, ROI_BRIGHT_PERCENTILE);

    cv::Mat norm8(centerFloat.size(), CV_8U, cv::Scalar(0));

    for (int y = 0; y < centerFloat.rows; ++y)
    {
        const float* src = centerFloat.ptr<float>(y);
        uchar* dst = norm8.ptr<uchar>(y);

        for (int x = 0; x < centerFloat.cols; ++x)
        {
            const float t = clamp01((src[x] - low) / range);
            dst[x] = cv::saturate_cast<uchar>(255.0f * t);
        }
    }

    cv::Mat otsuMask;
    const double otsu8 = cv::threshold(
        norm8,
        otsuMask,
        0,
        255,
        cv::THRESH_BINARY | cv::THRESH_OTSU
        );

    const float otsuThreshold =
        low + static_cast<float>(otsu8 / 255.0) * range;

    const float relaxedOtsu =
        low + 0.88f * (otsuThreshold - low);

    const float thresholdValue =
        std::max(relaxedOtsu, pThreshold);

    cv::Mat raw;
    cv::compare(centerFloat, thresholdValue, raw, cv::CMP_GE);

    const int closeDiameter = 2 * ROI_CLOSE_RADIUS + 1;
    const cv::Mat closeKernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(closeDiameter, closeDiameter)
        );

    cv::morphologyEx(
        raw,
        raw,
        cv::MORPH_CLOSE,
        closeKernel,
        cv::Point(-1, -1),
        2
        );

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;

    const int count = cv::connectedComponentsWithStats(
        raw,
        labels,
        stats,
        centroids,
        8,
        CV_32S
        );

    int largestLabel = -1;
    int largestArea = 0;

    for (int label = 1; label < count; ++label)
    {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);

        if (area >= ROI_MIN_COMPONENT_AREA && area > largestArea)
        {
            largestArea = area;
            largestLabel = label;
        }
    }

    cv::Mat mainMask(centerFloat.size(), CV_8U, cv::Scalar(0));

    if (largestLabel < 0)
    {
        // Broad ROI fails open, because it is only a post-filter.
        mainMask.setTo(255);

        // Internal persistent seeding must fail CLOSED. Otherwise a
        // failed intensity segmentation could promote ray artefacts
        // everywhere in the reconstruction field.
        if (coreOut != nullptr)
            *coreOut = cv::Mat(centerFloat.size(), CV_8U, cv::Scalar(0));

        return mainMask;
    }

    cv::compare(labels, cv::Scalar(largestLabel), mainMask, cv::CMP_EQ);
    mainMask = fillBinaryHoles(mainMask);

    if (coreOut != nullptr)
    {
        cv::Mat core = mainMask.clone();

        if (INTERNAL_CORE_ERODE_RADIUS > 0)
        {
            const int d = 2 * INTERNAL_CORE_ERODE_RADIUS + 1;
            const cv::Mat kernel = cv::getStructuringElement(
                cv::MORPH_ELLIPSE,
                cv::Size(d, d)
                );

            cv::erode(core, core, kernel);
        }

        *coreOut = core;
    }

    const int dilateDiameter = 2 * ROI_DILATE_RADIUS + 1;
    const cv::Mat dilateKernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(dilateDiameter, dilateDiameter)
        );

    cv::dilate(mainMask, mainMask, dilateKernel);

    return mainMask;
}

// ============================================================
// BROAD 3D ROI VOLUME
// ============================================================
// Build the same generous filled+dilated support independently for each
// test slice. This is an engineering acceleration/filter, not the surface
// locator: NMS still decides the edge position inside this broad support.
// If ROI extraction fails on a slice, buildBroadObjectROI fails open.
// ============================================================

static void buildROIVolumes(
    const std::vector<float>& volume,
    int width,
    int height,
    int depth,
    std::vector<uint8_t>& broadROI,
    std::vector<uint8_t>& coreROI)
{
    std::cout << "Building broad/core 3D object ROI..." << std::endl;

    const size_t sliceSize =
        static_cast<size_t>(width) *
        static_cast<size_t>(height);

    broadROI.assign(
        sliceSize * static_cast<size_t>(depth),
        0
        );

    coreROI.assign(
        sliceSize * static_cast<size_t>(depth),
        0
        );

    for (int z = 0; z < depth; ++z)
    {
        cv::Mat slice(
            height,
            width,
            CV_32F,
            const_cast<float*>(
                volume.data() + static_cast<size_t>(z) * sliceSize
                )
            );

        cv::Mat coreMask;
        const cv::Mat broadMask =
            buildBroadObjectROI(slice, &coreMask);

        uint8_t* broadDst =
            broadROI.data() + static_cast<size_t>(z) * sliceSize;

        uint8_t* coreDst =
            coreROI.data() + static_cast<size_t>(z) * sliceSize;

        for (int y = 0; y < height; ++y)
        {
            const uchar* broadSrc = broadMask.ptr<uchar>(y);
            const uchar* coreSrc = coreMask.ptr<uchar>(y);

            std::memcpy(
                broadDst + static_cast<size_t>(y) * width,
                broadSrc,
                static_cast<size_t>(width) * sizeof(uint8_t)
                );

            std::memcpy(
                coreDst + static_cast<size_t>(y) * width,
                coreSrc,
                static_cast<size_t>(width) * sizeof(uint8_t)
                );
        }

        std::cout << "\rROI slice "
                  << (FIRST_SLICE + z)
                  << " / " << LAST_SLICE
                  << std::flush;
    }

    std::cout << std::endl;
}

static std::vector<float> collectNmsGradientSamplesInROI(
    const GradientVolume& gradient,
    const std::vector<uint8_t>& nms,
    const std::vector<uint8_t>& roi)
{
    std::vector<float> samples;

    const size_t sliceSize =
        static_cast<size_t>(gradient.width) *
        static_cast<size_t>(gradient.height);

    for (int z = 1;
         z < gradient.depth - 1;
         z += STAT_SAMPLE_Z_STRIDE)
    {
        for (int y = 1;
             y < gradient.height - 1;
             y += STAT_SAMPLE_XY_STRIDE)
        {
            for (int x = 1;
                 x < gradient.width - 1;
                 x += STAT_SAMPLE_XY_STRIDE)
            {
                const size_t idx =
                    static_cast<size_t>(z) * sliceSize +
                    static_cast<size_t>(y) * gradient.width +
                    static_cast<size_t>(x);

                if (nms[idx] != 0 && roi[idx] != 0)
                    samples.push_back(gradient.data[idx].mag);
            }
        }
    }

    return samples;
}

// ============================================================
// CENTER-SLICE MASK / SCALAR VISUALISATION
// ============================================================

static cv::Mat centerMaskFromVolume(
    const std::vector<uint8_t>& volumeMask,
    int width,
    int height,
    int centerZ)
{
    cv::Mat out(height, width, CV_8U, cv::Scalar(0));

    const size_t sliceSize =
        static_cast<size_t>(width) *
        static_cast<size_t>(height);

    const uint8_t* src =
        volumeMask.data() +
        static_cast<size_t>(centerZ) * sliceSize;

    for (int y = 0; y < height; ++y)
    {
        uchar* dst = out.ptr<uchar>(y);

        for (int x = 0; x < width; ++x)
        {
            if (src[static_cast<size_t>(y) * width + x] != 0)
                dst[x] = 255;
        }
    }

    return out;
}

static cv::Mat scalarCenterImage(
    const std::vector<float>& values,
    int width,
    int height,
    int centerZ,
    float upperPercentile = 99.5f)
{
    const size_t sliceSize =
        static_cast<size_t>(width) *
        static_cast<size_t>(height);

    const size_t offset =
        static_cast<size_t>(centerZ) * sliceSize;

    std::vector<float> samples;
    samples.reserve(sliceSize / 4 + 1);

    for (int y = 0; y < height; y += 2)
    {
        for (int x = 0; x < width; x += 2)
        {
            const float v =
                values[offset + static_cast<size_t>(y) * width + x];

            if (v > 0.0f && std::isfinite(v))
                samples.push_back(v);
        }
    }

    const float hi =
        samples.empty()
            ? 1.0f
            : std::max(percentile(samples, upperPercentile), 1e-12f);

    cv::Mat out(height, width, CV_8U, cv::Scalar(0));

    for (int y = 0; y < height; ++y)
    {
        uchar* dst = out.ptr<uchar>(y);

        for (int x = 0; x < width; ++x)
        {
            const float v =
                values[offset + static_cast<size_t>(y) * width + x];

            dst[x] = cv::saturate_cast<uchar>(
                255.0f * clamp01(v / hi)
                );
        }
    }

    return out;
}

static cv::Mat zSupportCenterImage(
    const SurfaceMetricsVolume& metrics,
    int centerZ)
{
    cv::Mat out(metrics.height, metrics.width, CV_8U, cv::Scalar(0));

    const size_t sliceSize =
        static_cast<size_t>(metrics.width) *
        static_cast<size_t>(metrics.height);

    const size_t offset =
        static_cast<size_t>(centerZ) * sliceSize;

    for (int y = 0; y < metrics.height; ++y)
    {
        uchar* dst = out.ptr<uchar>(y);

        for (int x = 0; x < metrics.width; ++x)
        {
            const uint8_t hits =
                metrics.zSupport[
                    offset + static_cast<size_t>(y) * metrics.width + x
            ];

            dst[x] = cv::saturate_cast<uchar>(
                255.0f *
                static_cast<float>(hits) /
                static_cast<float>(2 * Z_SUPPORT_RADIUS)
                );
        }
    }

    return out;
}

static cv::Mat makeGradientCenterImage(
    const GradientVolume& gradient,
    int centerZ)
{
    cv::Mat values(
        gradient.height,
        gradient.width,
        CV_32F,
        cv::Scalar(0)
        );

    for (int y = 0; y < gradient.height; ++y)
    {
        float* dst = values.ptr<float>(y);

        for (int x = 0; x < gradient.width; ++x)
            dst[x] = gradient.at(x, y, centerZ).mag;
    }

    std::vector<float> samples;
    samples.reserve(
        static_cast<size_t>(gradient.width / 2 + 1) *
        static_cast<size_t>(gradient.height / 2 + 1)
        );

    for (int y = 0; y < gradient.height; y += 2)
    {
        const float* row = values.ptr<float>(y);
        for (int x = 0; x < gradient.width; x += 2)
            samples.push_back(row[x]);
    }

    const float hi = std::max(percentile(samples, 99.5f), 1e-12f);

    cv::Mat out(values.size(), CV_8U, cv::Scalar(0));

    for (int y = 0; y < values.rows; ++y)
    {
        const float* src = values.ptr<float>(y);
        uchar* dst = out.ptr<uchar>(y);

        for (int x = 0; x < values.cols; ++x)
            dst[x] = cv::saturate_cast<uchar>(255.0f * clamp01(src[x] / hi));
    }

    return out;
}

static cv::Mat makeNmsCenterImage(
    const GradientVolume& gradient,
    const std::vector<uint8_t>& nms,
    int centerZ)
{
    cv::Mat values(
        gradient.height,
        gradient.width,
        CV_32F,
        cv::Scalar(0)
        );

    const size_t sliceSize =
        static_cast<size_t>(gradient.width) *
        static_cast<size_t>(gradient.height);

    for (int y = 0; y < gradient.height; ++y)
    {
        float* dst = values.ptr<float>(y);

        for (int x = 0; x < gradient.width; ++x)
        {
            const size_t idx =
                static_cast<size_t>(centerZ) * sliceSize +
                static_cast<size_t>(y) * gradient.width + x;

            if (nms[idx] != 0)
                dst[x] = gradient.data[idx].mag;
        }
    }

    std::vector<float> samples;
    samples.reserve(static_cast<size_t>(gradient.width * gradient.height / 4));

    for (int y = 0; y < values.rows; y += 2)
    {
        const float* row = values.ptr<float>(y);
        for (int x = 0; x < values.cols; x += 2)
        {
            if (row[x] > 0.0f)
                samples.push_back(row[x]);
        }
    }

    const float hi =
        samples.empty() ? 1.0f : std::max(percentile(samples, 99.0f), 1e-12f);

    cv::Mat out(values.size(), CV_8U, cv::Scalar(0));

    for (int y = 0; y < values.rows; ++y)
    {
        const float* src = values.ptr<float>(y);
        uchar* dst = out.ptr<uchar>(y);

        for (int x = 0; x < values.cols; ++x)
            dst[x] = cv::saturate_cast<uchar>(255.0f * clamp01(src[x] / hi));
    }

    return out;
}

// ============================================================
// TAYLOR SUBVOXEL REFINEMENT - ARTICLE EQUATIONS
// ============================================================

static bool refineTaylorPoint(
    const std::vector<float>& volume,
    int width,
    int height,
    int depth,
    const GradientVolume& gradient,
    const SurfaceMetricsVolume& metrics,
    int x,
    int y,
    int z,
    SubvoxelPoint& out)
{
    // Third derivatives require a two-voxel margin.
    if (x < 2 || x >= width - 2 ||
        y < 2 || y >= height - 2 ||
        z < 2 || z >= depth - 2)
    {
        return false;
    }

    const GradientVoxel& g = gradient.at(x, y, z);
    if (g.mag <= 1e-12f)
        return false;

    const cv::Vec3f n = normalFromGradient(g);

    const double nx = n[0];
    const double ny = n[1];
    const double nz = n[2];

    const double hx = SPACING_X;
    const double hy = SPACING_Y;
    const double hz = SPACING_Z;

    auto f = [&](int xx, int yy, int zz) -> double
    {
        return static_cast<double>(
            voxelValue(volume, width, height, depth, xx, yy, zz)
            );
    };

    const double f0 = f(x, y, z);

    // -------------------- second partials --------------------
    const double fxx =
        (f(x + 1, y, z) - 2.0 * f0 + f(x - 1, y, z)) /
        (hx * hx);

    const double fyy =
        (f(x, y + 1, z) - 2.0 * f0 + f(x, y - 1, z)) /
        (hy * hy);

    const double fzz =
        (f(x, y, z + 1) - 2.0 * f0 + f(x, y, z - 1)) /
        (hz * hz);

    const double fxy =
        (f(x + 1, y + 1, z) -
         f(x + 1, y - 1, z) -
         f(x - 1, y + 1, z) +
         f(x - 1, y - 1, z)) /
        (4.0 * hx * hy);

    const double fxz =
        (f(x + 1, y, z + 1) -
         f(x + 1, y, z - 1) -
         f(x - 1, y, z + 1) +
         f(x - 1, y, z - 1)) /
        (4.0 * hx * hz);

    const double fyz =
        (f(x, y + 1, z + 1) -
         f(x, y + 1, z - 1) -
         f(x, y - 1, z + 1) +
         f(x, y - 1, z - 1)) /
        (4.0 * hy * hz);

    const double d2 =
        nx * nx * fxx +
        2.0 * nx * ny * fxy +
        2.0 * nx * nz * fxz +
        ny * ny * fyy +
        2.0 * ny * nz * fyz +
        nz * nz * fzz;

    // -------------------- third partials ---------------------
    const double fxxx =
        (f(x + 2, y, z) -
         2.0 * f(x + 1, y, z) +
         2.0 * f(x - 1, y, z) -
         f(x - 2, y, z)) /
        (2.0 * hx * hx * hx);

    const double fyyy =
        (f(x, y + 2, z) -
         2.0 * f(x, y + 1, z) +
         2.0 * f(x, y - 1, z) -
         f(x, y - 2, z)) /
        (2.0 * hy * hy * hy);

    const double fzzz =
        (f(x, y, z + 2) -
         2.0 * f(x, y, z + 1) +
         2.0 * f(x, y, z - 1) -
         f(x, y, z - 2)) /
        (2.0 * hz * hz * hz);

    auto fxxAt = [&](int xx, int yy, int zz) -> double
    {
        const double c = f(xx, yy, zz);
        return
            (f(xx + 1, yy, zz) - 2.0 * c + f(xx - 1, yy, zz)) /
            (hx * hx);
    };

    auto fyyAt = [&](int xx, int yy, int zz) -> double
    {
        const double c = f(xx, yy, zz);
        return
            (f(xx, yy + 1, zz) - 2.0 * c + f(xx, yy - 1, zz)) /
            (hy * hy);
    };

    auto fzzAt = [&](int xx, int yy, int zz) -> double
    {
        const double c = f(xx, yy, zz);
        return
            (f(xx, yy, zz + 1) - 2.0 * c + f(xx, yy, zz - 1)) /
            (hz * hz);
    };

    const double fxxy =
        (fxxAt(x, y + 1, z) - fxxAt(x, y - 1, z)) /
        (2.0 * hy);

    const double fxxz =
        (fxxAt(x, y, z + 1) - fxxAt(x, y, z - 1)) /
        (2.0 * hz);

    const double fxyy =
        (fyyAt(x + 1, y, z) - fyyAt(x - 1, y, z)) /
        (2.0 * hx);

    const double fxzz =
        (fzzAt(x + 1, y, z) - fzzAt(x - 1, y, z)) /
        (2.0 * hx);

    const double fyyz =
        (fyyAt(x, y, z + 1) - fyyAt(x, y, z - 1)) /
        (2.0 * hz);

    const double fyzz =
        (fzzAt(x, y + 1, z) - fzzAt(x, y - 1, z)) /
        (2.0 * hy);

    const double fxyz =
        (
            f(x + 1, y + 1, z + 1) -
            f(x + 1, y + 1, z - 1) -
            f(x + 1, y - 1, z + 1) +
            f(x + 1, y - 1, z - 1) -
            f(x - 1, y + 1, z + 1) +
            f(x - 1, y + 1, z - 1) +
            f(x - 1, y - 1, z + 1) -
            f(x - 1, y - 1, z - 1)
            ) /
        (8.0 * hx * hy * hz);

    const double d3 =
        nx * nx * nx * fxxx +
        3.0 * nx * nx * ny * fxxy +
        3.0 * nx * nx * nz * fxxz +
        3.0 * nx * ny * ny * fxyy +
        6.0 * nx * ny * nz * fxyz +
        3.0 * nx * nz * nz * fxzz +
        ny * ny * ny * fyyy +
        3.0 * ny * ny * nz * fyyz +
        3.0 * ny * nz * nz * fyzz +
        nz * nz * nz * fzzz;

    if (!std::isfinite(d2) ||
        !std::isfinite(d3) ||
        std::abs(d3) < TAYLOR_MIN_ABS_D3)
    {
        return false;
    }

    // At the maximum of the first directional derivative, D3 f < 0.
    if (TAYLOR_REQUIRE_LOCAL_MAXIMUM && d3 >= 0.0)
        return false;

    const double tPhysical = -d2 / d3; // micrometres

    if (!std::isfinite(tPhysical))
        return false;

    const double maxPhysical =
        static_cast<double>(TAYLOR_MAX_ABS_DISPLACEMENT_VOXELS) *
        static_cast<double>(
            std::min({SPACING_X, SPACING_Y, SPACING_Z})
            );

    if (std::abs(tPhysical) > maxPhysical)
        return false;

    out.sourceX = x;
    out.sourceY = y;
    out.sourceZ = z;
    out.tPhysicalUm = static_cast<float>(tPhysical);

    out.x =
        static_cast<float>(
            static_cast<double>(x) + nx * tPhysical / SPACING_X
            );

    out.y =
        static_cast<float>(
            static_cast<double>(y) + ny * tPhysical / SPACING_Y
            );

    out.z =
        static_cast<float>(
            static_cast<double>(z) + nz * tPhysical / SPACING_Z
            );

    out.physicalX = out.x * SPACING_X;
    out.physicalY = out.y * SPACING_Y;
    out.physicalZ =
        (static_cast<float>(FIRST_SLICE) + out.z) * SPACING_Z;

    out.gradientMagnitude = g.mag;

    const size_t idx = metrics.index(x, y, z);
    out.coherence = metrics.coherence[idx];
    out.contrast = metrics.contrast[idx];
    out.zSupport = metrics.zSupport[idx];
    out.surfaceScore = metrics.score[idx];
    out.refinementType = 1;

    return true;
}


// ============================================================
// SUBVOXEL FALLBACKS
// ============================================================
// The article's Taylor refinement is our preferred position estimator.
// Numerically, however, D_g^3 can be close to zero or have an unstable
// sign at a noisy yet otherwise valid Canny point. Dropping that point
// would re-open a contour. Therefore, for topology preservation only,
// use a local parabola of |grad f| along the same 3D normal. If that is
// also unstable, retain the accepted voxel centre.
// ============================================================

static void fillPointMetadata(
    const GradientVolume& gradient,
    const SurfaceMetricsVolume& metrics,
    int x,
    int y,
    int z,
    SubvoxelPoint& out)
{
    const GradientVoxel& g = gradient.at(x, y, z);
    const size_t idx = metrics.index(x, y, z);

    out.sourceX = x;
    out.sourceY = y;
    out.sourceZ = z;

    out.gradientMagnitude = g.mag;
    out.coherence = metrics.coherence[idx];
    out.contrast = metrics.contrast[idx];
    out.zSupport = metrics.zSupport[idx];
    out.surfaceScore = metrics.score[idx];
}

static bool refineParabolicGradientPoint(
    const GradientVolume& gradient,
    const SurfaceMetricsVolume& metrics,
    int x,
    int y,
    int z,
    SubvoxelPoint& out)
{
    if (x < 1 || x >= gradient.width - 1 ||
        y < 1 || y >= gradient.height - 1 ||
        z < 1 || z >= gradient.depth - 1)
    {
        return false;
    }

    const GradientVoxel& g = gradient.at(x, y, z);
    if (g.mag <= 1e-12f)
        return false;

    const cv::Vec3f normal = normalFromGradient(g);
    const cv::Vec3f d = normalVoxelDirection(normal);

    const float m0 = g.mag;

    const float mPlus =
        sampleMagnitude(
            gradient,
            static_cast<float>(x) + d[0],
            static_cast<float>(y) + d[1],
            static_cast<float>(z) + d[2]
            );

    const float mMinus =
        sampleMagnitude(
            gradient,
            static_cast<float>(x) - d[0],
            static_cast<float>(y) - d[1],
            static_cast<float>(z) - d[2]
            );

    const float denominator =
        mMinus - 2.0f * m0 + mPlus;

    // Concave parabola -> local maximum.
    if (!std::isfinite(denominator) ||
        denominator >= -1e-12f)
    {
        return false;
    }

    float delta =
        0.5f * (mMinus - mPlus) / denominator;

    if (!std::isfinite(delta))
        return false;

    delta =
        std::clamp(
            delta,
            -PARABOLIC_MAX_ABS_OFFSET,
            PARABOLIC_MAX_ABS_OFFSET
            );

    fillPointMetadata(
        gradient,
        metrics,
        x, y, z,
        out
        );

    out.x = static_cast<float>(x) + delta * d[0];
    out.y = static_cast<float>(y) + delta * d[1];
    out.z = static_cast<float>(z) + delta * d[2];

    const float physicalStep =
        std::sqrt(
            (d[0] * SPACING_X) * (d[0] * SPACING_X) +
            (d[1] * SPACING_Y) * (d[1] * SPACING_Y) +
            (d[2] * SPACING_Z) * (d[2] * SPACING_Z)
            );

    out.tPhysicalUm = delta * physicalStep;

    out.physicalX = out.x * SPACING_X;
    out.physicalY = out.y * SPACING_Y;
    out.physicalZ =
        (static_cast<float>(FIRST_SLICE) + out.z) * SPACING_Z;

    out.refinementType = 2;

    return true;
}

static void keepVoxelCenterPoint(
    const GradientVolume& gradient,
    const SurfaceMetricsVolume& metrics,
    int x,
    int y,
    int z,
    SubvoxelPoint& out)
{
    fillPointMetadata(
        gradient,
        metrics,
        x, y, z,
        out
        );

    out.tPhysicalUm = 0.0f;

    out.x = static_cast<float>(x);
    out.y = static_cast<float>(y);
    out.z = static_cast<float>(z);

    out.physicalX = out.x * SPACING_X;
    out.physicalY = out.y * SPACING_Y;
    out.physicalZ =
        (static_cast<float>(FIRST_SLICE) + out.z) * SPACING_Z;

    out.refinementType = 3;
}

// ============================================================
// OPTIONAL SAVE OF ALL ACCEPTED TEST-WINDOW MASKS
// ============================================================

static void saveAcceptedSlices(
    const std::vector<uint8_t>& accepted,
    int width,
    int height,
    int depth)
{
    if (!SAVE_ACCEPTED_MASK_FOR_ALL_TEST_SLICES)
        return;

    const fs::path folder =
        fs::path(OUTPUT_FOLDER) / "accepted_slices";

    fs::create_directories(folder);

    const size_t sliceSize =
        static_cast<size_t>(width) *
        static_cast<size_t>(height);

    for (int z = 0; z < depth; ++z)
    {
        cv::Mat mask(height, width, CV_8U, cv::Scalar(0));

        const uint8_t* src =
            accepted.data() +
            static_cast<size_t>(z) * sliceSize;

        for (int y = 0; y < height; ++y)
        {
            uchar* dst = mask.ptr<uchar>(y);

            for (int x = 0; x < width; ++x)
            {
                if (src[static_cast<size_t>(y) * width + x] != 0)
                    dst[x] = 255;
            }
        }

        std::ostringstream name;
        name << "accepted_"
             << std::setw(4)
             << std::setfill('0')
             << (FIRST_SLICE + z)
             << ".png";

        cv::imwrite((folder / name.str()).string(), mask);
    }
}

// ============================================================
// MAIN
// ============================================================

int main()
{
    try
    {
        fs::create_directories(OUTPUT_FOLDER);

        const std::vector<fs::path> files = collectTiffFiles(INPUT_FOLDER);

        if (files.empty())
        {
            std::cerr << "No TIFF files found." << std::endl;
            return 1;
        }

        if (FIRST_SLICE < 0 ||
            LAST_SLICE < FIRST_SLICE ||
            CENTER_SLICE < FIRST_SLICE ||
            CENTER_SLICE > LAST_SLICE ||
            static_cast<size_t>(LAST_SLICE) >= files.size())
        {
            std::cerr << "Invalid slice indices. TIFF count = "
                      << files.size() << std::endl;
            return 1;
        }

        const int centerZ = CENTER_SLICE - FIRST_SLICE;

        cv::Mat centerOriginal;
        if (!loadFloatSlice(files[CENTER_SLICE], centerOriginal))
            return 1;

        int width = 0;
        int height = 0;
        int depth = 0;

        std::vector<float> volume;

        if (!loadVolumeXYSmoothed(
                files,
                volume,
                width,
                height,
                depth))
        {
            return 1;
        }

        gaussianBlurZInPlace(volume, width, height, depth);

        // ----------------------------------------------------
        // ARTICLE CORE: 3D GRADIENT -> 3D NMS
        // ----------------------------------------------------

        GradientVolume gradient =
            computeGradient3D(volume, width, height, depth);

        std::vector<float> gradientSamples;
        std::vector<uint8_t> nms =
            computeNMS3D(gradient, gradientSamples);

        if (gradientSamples.empty())
        {
            std::cerr << "3D NMS produced no candidates." << std::endl;
            return 1;
        }

        // ----------------------------------------------------
        // BROAD 3D SUPPORT ROI
        // ----------------------------------------------------
        // It is intentionally filled and dilated. It does NOT locate
        // the surface. It only removes remote reconstruction field
        // from the expensive quality calculations/hysteresis.

        std::vector<uint8_t> broadROI3D;
        std::vector<uint8_t> coreROI3D;

        buildROIVolumes(
            volume,
            width,
            height,
            depth,
            broadROI3D,
            coreROI3D
            );

        gradientSamples =
            collectNmsGradientSamplesInROI(
                gradient,
                nms,
                broadROI3D
                );

        if (gradientSamples.empty())
        {
            std::cerr << "Broad ROI contains no NMS candidates." << std::endl;
            return 1;
        }

        // ----------------------------------------------------
        // OUR ADDITIONS: 3D COHERENCE + NORMAL CONTRAST + Z SUPPORT
        // ----------------------------------------------------

        std::vector<float> scoreSamples;

        SurfaceMetricsVolume metrics =
            computeSurfaceMetrics(
                volume,
                width,
                height,
                depth,
                gradient,
                nms,
                broadROI3D,
                gradientSamples,
                scoreSamples
                );

        if (scoreSamples.empty())
        {
            std::cerr << "No surface-score candidates." << std::endl;
            return 1;
        }

        const float strongThreshold =
            percentile(scoreSamples, STRONG_SCORE_PERCENTILE);

        const float weakPercentileThreshold =
            percentile(scoreSamples, WEAK_SCORE_PERCENTILE);

        const float weakThreshold =
            std::max(
                weakPercentileThreshold,
                strongThreshold * WEAK_TO_STRONG_RATIO
                );

        std::cout
            << "\nSurface-score thresholds:\n"
            << "  weak percentile threshold = "
            << weakPercentileThreshold << '\n'
            << "  weak final threshold      = "
            << weakThreshold << '\n'
            << "  strong threshold          = "
            << strongThreshold << '\n'
            << "  rescue threshold          = "
            << weakThreshold * RESCUE_TO_WEAK_RATIO
            << std::endl;

        // ----------------------------------------------------
        // ARTICLE CORE: 3D HYSTERESIS, enriched by local 3D quality
        // ----------------------------------------------------

        std::vector<uint8_t> edgeType;
        std::vector<uint8_t> accepted3D;

        hysteresis3D(
            gradient,
            nms,
            metrics,
            coreROI3D,
            weakThreshold,
            strongThreshold,
            edgeType,
            accepted3D
            );

        const cv::Mat hysteresisBeforeBridge =
            centerMaskFromVolume(
                accepted3D,
                width,
                height,
                centerZ
                );

        const long long bridgeAdded3D =
            bridgeShortContourGaps3D(
                gradient,
                nms,
                metrics,
                coreROI3D,
                weakThreshold,
                edgeType,
                accepted3D
                );

        // ----------------------------------------------------
        // CENTER-SLICE DIAGNOSTICS BEFORE ROI
        // ----------------------------------------------------

        const cv::Mat gradientCenter =
            makeGradientCenterImage(gradient, centerZ);

        const cv::Mat nmsCenter =
            makeNmsCenterImage(gradient, nms, centerZ);

        const cv::Mat coherenceCenter =
            scalarCenterImage(
                metrics.coherence,
                width,
                height,
                centerZ,
                99.8f
                );

        const cv::Mat contrastCenter =
            scalarCenterImage(
                metrics.contrastNorm,
                width,
                height,
                centerZ,
                99.8f
                );

        const cv::Mat zSupportCenter =
            zSupportCenterImage(metrics, centerZ);

        const cv::Mat scoreCenter =
            scalarCenterImage(
                metrics.score,
                width,
                height,
                centerZ,
                99.5f
                );

        const cv::Mat hysteresisCenterUnfiltered =
            centerMaskFromVolume(
                accepted3D,
                width,
                height,
                centerZ
                );

        // ----------------------------------------------------
        // GENEROUS POST-DETECTION ROI
        // ----------------------------------------------------

        const cv::Mat broadROI =
            centerMaskFromVolume(
                broadROI3D,
                width,
                height,
                centerZ
                );

        const cv::Mat coreROI =
            centerMaskFromVolume(
                coreROI3D,
                width,
                height,
                centerZ
                );

        cv::Mat finalCenter;
        cv::bitwise_and(
            hysteresisCenterUnfiltered,
            broadROI,
            finalCenter
            );

        // ----------------------------------------------------
        // CLASSIFICATION IMAGE
        // ----------------------------------------------------

        cv::Mat classification(
            height,
            width,
            CV_8UC3,
            cv::Scalar(0, 0, 0)
            );

        const size_t sliceSize =
            static_cast<size_t>(width) *
            static_cast<size_t>(height);

        const size_t centerOffset =
            static_cast<size_t>(centerZ) * sliceSize;

        long long centerAccepted = 0;
        long long centerStrong = 0;
        long long centerPersistentSeed = 0;
        long long centerWeak = 0;
        long long centerRescue = 0;
        long long centerBridge = 0;

        for (int y = 0; y < height; ++y)
        {
            const uchar* finalRow = finalCenter.ptr<uchar>(y);
            cv::Vec3b* dst = classification.ptr<cv::Vec3b>(y);

            for (int x = 0; x < width; ++x)
            {
                if (finalRow[x] == 0)
                    continue;

                const size_t idx =
                    centerOffset +
                    static_cast<size_t>(y) * width +
                    static_cast<size_t>(x);

                ++centerAccepted;

                if (edgeType[idx] == 2)
                {
                    // global strong = red
                    dst[x] = cv::Vec3b(0, 0, 255);
                    ++centerStrong;
                }
                else if (edgeType[idx] == 4)
                {
                    // persistent internal 3D seed = green
                    dst[x] = cv::Vec3b(0, 255, 0);
                    ++centerPersistentSeed;
                }
                else if (edgeType[idx] == 5)
                {
                    // evidence-guided short-gap bridge = blue
                    dst[x] = cv::Vec3b(255, 0, 0);
                    ++centerBridge;
                }
                else if (edgeType[idx] == 3)
                {
                    // 3D rescue connected to a seed = magenta
                    dst[x] = cv::Vec3b(255, 0, 255);
                    ++centerRescue;
                }
                else
                {
                    // ordinary weak connected = yellow
                    dst[x] = cv::Vec3b(0, 255, 255);
                    ++centerWeak;
                }
            }
        }

        // ----------------------------------------------------
        // ARTICLE CORE: THIRD-ORDER TAYLOR SUBVOXEL REFINEMENT
        // ----------------------------------------------------

        std::vector<SubvoxelPoint> refinedPoints;
        refinedPoints.reserve(
            static_cast<size_t>(std::max<long long>(centerAccepted, 0))
            );

        cv::Mat subvoxelMask(
            height,
            width,
            CV_8U,
            cv::Scalar(0)
            );

        long long taylorPrimary = 0;
        long long taylorPrimaryFailed = 0;
        long long parabolicFallback = 0;
        long long voxelCenterFallback = 0;

        for (int y = 2; y < height - 2; ++y)
        {
            const uchar* finalRow = finalCenter.ptr<uchar>(y);

            for (int x = 2; x < width - 2; ++x)
            {
                if (finalRow[x] == 0)
                    continue;

                SubvoxelPoint point;

                bool refined =
                    refineTaylorPoint(
                        volume,
                        width,
                        height,
                        depth,
                        gradient,
                        metrics,
                        x,
                        y,
                        centerZ,
                        point
                        );

                if (refined)
                {
                    ++taylorPrimary;
                }
                else
                {
                    ++taylorPrimaryFailed;

                    if (!ENABLE_SUBVOXEL_FALLBACK)
                        continue;

                    refined =
                        refineParabolicGradientPoint(
                            gradient,
                            metrics,
                            x,
                            y,
                            centerZ,
                            point
                            );

                    if (refined)
                    {
                        ++parabolicFallback;
                    }
                    else
                    {
                        keepVoxelCenterPoint(
                            gradient,
                            metrics,
                            x,
                            y,
                            centerZ,
                            point
                            );

                        refined = true;
                        ++voxelCenterFallback;
                    }
                }

                refinedPoints.push_back(point);

                const int xx = cvRound(point.x);
                const int yy = cvRound(point.y);

                if (xx >= 0 && xx < width &&
                    yy >= 0 && yy < height)
                {
                    subvoxelMask.at<uchar>(yy, xx) = 255;
                }
            }
        }

        // ----------------------------------------------------
        // OVERLAYS
        // ----------------------------------------------------

        const cv::Mat display = makeDisplayImage(centerOriginal);

        cv::Mat overlay;
        cv::cvtColor(display, overlay, cv::COLOR_GRAY2BGR);

        for (int y = 0; y < height; ++y)
        {
            const cv::Vec3b* cls = classification.ptr<cv::Vec3b>(y);
            cv::Vec3b* dst = overlay.ptr<cv::Vec3b>(y);

            for (int x = 0; x < width; ++x)
            {
                if (cls[x][0] != 0 || cls[x][1] != 0 || cls[x][2] != 0)
                    dst[x] = cls[x];
            }
        }

        cv::Mat subvoxelOverlay = overlay.clone();

        for (const SubvoxelPoint& p : refinedPoints)
        {
            const int xx = cvRound(p.x);
            const int yy = cvRound(p.y);

            if (xx >= 0 && xx < width &&
                yy >= 0 && yy < height)
            {
                // final refined/fallback surface point = cyan
                subvoxelOverlay.at<cv::Vec3b>(yy, xx) =
                    cv::Vec3b(255, 255, 0);
            }
        }

        // ----------------------------------------------------
        // SAVE DIAGNOSTICS
        // ----------------------------------------------------

        cv::imwrite(
            OUTPUT_FOLDER + "/00_gradient_3D_center.png",
            gradientCenter
            );

        cv::imwrite(
            OUTPUT_FOLDER + "/01_NMS_3D_center.png",
            nmsCenter
            );

        cv::imwrite(
            OUTPUT_FOLDER + "/02_gradient_coherence_center.png",
            coherenceCenter
            );

        cv::imwrite(
            OUTPUT_FOLDER + "/03_normal_contrast_center.png",
            contrastCenter
            );

        cv::imwrite(
            OUTPUT_FOLDER + "/04_Z_support_center.png",
            zSupportCenter
            );

        cv::imwrite(
            OUTPUT_FOLDER + "/05_surface_score_center.png",
            scoreCenter
            );

        cv::imwrite(
            OUTPUT_FOLDER + "/06_hysteresis_before_bridge.png",
            hysteresisBeforeBridge
            );

        cv::imwrite(
            OUTPUT_FOLDER + "/07_hysteresis_after_bridge.png",
            hysteresisCenterUnfiltered
            );

        cv::imwrite(
            OUTPUT_FOLDER + "/08_object_core_ROI.png",
            coreROI
            );

        cv::imwrite(
            OUTPUT_FOLDER + "/09_broad_object_ROI.png",
            broadROI
            );

        cv::imwrite(
            OUTPUT_FOLDER + "/10_final_center_mask.png",
            finalCenter
            );

        cv::imwrite(
            OUTPUT_FOLDER + "/11_final_center_classification.png",
            classification
            );

        cv::imwrite(
            OUTPUT_FOLDER + "/12_overlay.png",
            overlay
            );

        cv::imwrite(
            OUTPUT_FOLDER + "/13_subvoxel_surface_mask.png",
            subvoxelMask
            );

        cv::imwrite(
            OUTPUT_FOLDER + "/14_subvoxel_surface_overlay.png",
            subvoxelOverlay
            );

        saveAcceptedSlices(accepted3D, width, height, depth);

        // ----------------------------------------------------
        // SAVE SUBVOXEL POINT CLOUD FOR CENTER-SLICE SEEDS
        // ----------------------------------------------------

        std::ofstream csv(
            OUTPUT_FOLDER + "/surface_points_center_refined.csv"
            );

        csv << std::fixed << std::setprecision(9);
        csv
            << "source_x,source_y,source_local_z,source_global_z,"
               "t_um,x_voxel,y_voxel,z_local_voxel,"
               "x_um,y_um,z_um,gradient_magnitude,"
               "coherence,normal_contrast,z_support,surface_score,"
               "refinement_type\n";

        for (const SubvoxelPoint& p : refinedPoints)
        {
            csv
                << p.sourceX << ','
                << p.sourceY << ','
                << p.sourceZ << ','
                << (FIRST_SLICE + p.sourceZ) << ','
                << p.tPhysicalUm << ','
                << p.x << ','
                << p.y << ','
                << p.z << ','
                << p.physicalX << ','
                << p.physicalY << ','
                << p.physicalZ << ','
                << p.gradientMagnitude << ','
                << p.coherence << ','
                << p.contrast << ','
                << p.zSupport << ','
                << p.surfaceScore << ','
                << p.refinementType
                << '\n';
        }

        csv.close();

        // ----------------------------------------------------
        // REPORT
        // ----------------------------------------------------

        const long long totalNms =
            static_cast<long long>(
                std::count(
                    nms.begin(),
                    nms.end(),
                    static_cast<uint8_t>(1)
                    )
                );

        const long long totalAccepted3D =
            static_cast<long long>(
                std::count(
                    accepted3D.begin(),
                    accepted3D.end(),
                    static_cast<uint8_t>(1)
                    )
                );

        std::ofstream report(
            OUTPUT_FOLDER + "/report_article_surface_v4.txt"
            );

        report << std::fixed << std::setprecision(9);

        report
            << "ARTICLE-BASED 3D CANNY + MOTION-TOLERANT Z TRACKING + BRIDGING + TAYLOR\n"
            << "=================================================================\n\n"
            << "Slices\n"
            << "------\n"
            << "first  = " << FIRST_SLICE << '\n'
            << "center = " << CENTER_SLICE << '\n'
            << "last   = " << LAST_SLICE << '\n'
            << "depth  = " << depth << "\n\n"
            << "Voxel spacing (um)\n"
            << "------------------\n"
            << "sx = " << SPACING_X << '\n'
            << "sy = " << SPACING_Y << '\n'
            << "sz = " << SPACING_Z << "\n\n"
            << "Gaussian\n"
            << "--------\n"
            << "sigma_xy = " << SIGMA_XY << '\n'
            << "sigma_z  = " << SIGMA_Z << "\n\n"
            << "3D NMS\n"
            << "------\n"
            << "step = " << NMS_STEP << '\n'
            << "plateau factor = " << NMS_PLATEAU_FACTOR << '\n'
            << "NMS voxels = " << totalNms << "\n\n"
            << "Quality metric scales\n"
            << "---------------------\n"
            << "gradient scale P" << GRADIENT_SCALE_PERCENTILE
            << " = " << metrics.gradientScale << '\n'
            << "contrast scale P" << CONTRAST_SCALE_PERCENTILE
            << " = " << metrics.contrastScale << "\n\n"
            << "Hysteresis thresholds on composite score\n"
            << "----------------------------------------\n"
            << "weak percentile = " << WEAK_SCORE_PERCENTILE << '\n'
            << "strong percentile = " << STRONG_SCORE_PERCENTILE << '\n'
            << "weak/strong ratio floor = " << WEAK_TO_STRONG_RATIO << '\n'
            << "weak percentile threshold = "
            << weakPercentileThreshold << '\n'
            << "weak final threshold = " << weakThreshold << '\n'
            << "strong threshold = " << strongThreshold << '\n'
            << "rescue threshold = "
            << weakThreshold * RESCUE_TO_WEAK_RATIO << '\n'
            << "persistent seed threshold = "
            << weakThreshold * PERSISTENT_SEED_TO_WEAK_RATIO << "\n\n"
            << "3D persistence / bridge parameters\n"
            << "----------------------------------\n"
            << "Z support radius = " << Z_SUPPORT_RADIUS << '\n'
            << "Z support search radius = " << Z_SUPPORT_SEARCH_RADIUS << '\n'
            << "Z support recovery radius = " << Z_SUPPORT_RECOVERY_SEARCH_RADIUS << '\n'
            << "Z support max consecutive misses = "
            << Z_SUPPORT_MAX_CONSECUTIVE_MISSES << '\n'
            << "Z support local normal cosine = "
            << Z_SUPPORT_MIN_NORMAL_COSINE << '\n'
            << "Z support seed normal cosine = "
            << Z_SUPPORT_MIN_SEED_NORMAL_COSINE << '\n'
            << "hysteresis track search radius = "
            << HYSTERESIS_TRACK_SEARCH_RADIUS << '\n'
            << "hysteresis track min normal cosine = "
            << HYSTERESIS_TRACK_MIN_NORMAL_COSINE << '\n'
            << "persistent seed min Z support = "
            << PERSISTENT_SEED_MIN_Z_SUPPORT << '\n'
            << "bridge min Z support = " << BRIDGE_MIN_Z_SUPPORT << '\n'
            << "bridge tangent radius = " << BRIDGE_TANGENT_RADIUS << "\n\n"
            << "Center results\n"
            << "--------------\n"
            << "accepted after broad ROI = " << centerAccepted << '\n'
            << "global strong = " << centerStrong << '\n'
            << "persistent 3D seeds = " << centerPersistentSeed << '\n'
            << "weak connected = " << centerWeak << '\n'
            << "3D rescue connected = " << centerRescue << '\n'
            << "short-gap bridge = " << centerBridge << "\n\n"
            << "Subvoxel results\n"
            << "----------------\n"
            << "Taylor primary success = " << taylorPrimary << '\n'
            << "Taylor primary failed = " << taylorPrimaryFailed << '\n'
            << "parabolic fallback = " << parabolicFallback << '\n'
            << "voxel-centre fallback = " << voxelCenterFallback << '\n'
            << "final refined/fallback points = " << refinedPoints.size()
            << "\n\n"
            << "Volume results\n"
            << "--------------\n"
            << "evidence-guided bridge points added = " << bridgeAdded3D << '\n'
            << "3D accepted voxels = " << totalAccepted3D << '\n';

        report.close();

        // ----------------------------------------------------
        // DONE
        // ----------------------------------------------------

        std::cout
            << "\n==============================================\n"
            << "DONE - ARTICLE SURFACE V4\n"
            << "==============================================\n"
            << "3D NMS voxels: " << totalNms << '\n'
            << "3D bridge points added: " << bridgeAdded3D << '\n'
            << "3D accepted: " << totalAccepted3D << '\n'
            << "Center after broad ROI: " << centerAccepted << '\n'
            << "  global strong: " << centerStrong << '\n'
            << "  persistent 3D seeds: " << centerPersistentSeed << '\n'
            << "  weak connected: " << centerWeak << '\n'
            << "  rescue connected: " << centerRescue << '\n'
            << "  short-gap bridge: " << centerBridge << '\n'
            << "Taylor primary success: " << taylorPrimary << '\n'
            << "Taylor primary failed: " << taylorPrimaryFailed << '\n'
            << "Parabolic fallback: " << parabolicFallback << '\n'
            << "Voxel-centre fallback: " << voxelCenterFallback << '\n'
            << "Final surface points: " << refinedPoints.size() << '\n'
            << "Results folder:\n"
            << OUTPUT_FOLDER << std::endl;

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
