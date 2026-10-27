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
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ============================================================
// CT EDGE-DETECTOR COMPARISON
// ============================================================
// This diagnostic program deliberately compares ONLY the detection stage:
//
//   1) ordinary/global-threshold 3D Canny;
//   2) adaptive/noise-aware 3D Canny;
//   3) noise-compensated phase congruency (2D Log-Gabor per slice)
//      followed by 3D hysteresis through the same slice stack.
//
// Taylor subvoxel refinement from the previous article-based program is
// intentionally NOT applied here. Otherwise detector quality and refinement
// quality would be mixed together. After we choose the most promising
// detector (or a hybrid), Taylor refinement can be attached again.
// ============================================================

// ============================================================
// PATHS / SAME TEST WINDOW AS THE PREVIOUS PROGRAM
// ============================================================

static const std::string INPUT_FOLDER =
    "D:/Studies/Real_work/BladeCT/"
    "20251010_Blade_16bit_75.40um_1804x910x1903";

static const std::string OUTPUT_FOLDER =
    "D:/Studies/Real_work/BladeCT/Results/three_detectors_comparison_v2";

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
// COMMON PREPROCESSING FOR CANNY
// ============================================================

constexpr double SIGMA_XY = 1.15;
constexpr double SIGMA_Z  = 1.00;
constexpr int Z_GAUSS_RADIUS = 3;

// ============================================================
// BROAD OBJECT ROI
// ============================================================
// This support mask is deliberately NOT an edge detector.  In V2 the seed
// is obtained with Otsu thresholding on a robustly normalised CT slice,
// followed by dominant-component selection, hole filling and dilation.
//
// Why this change:
//   * the previous percentile mask admitted large portions of reconstruction
//     texture, so adaptive Canny and phase congruency could seed on background;
//   * Otsu is used only to obtain a generous object support region, not to
//     define the measured surface itself;
//   * strong hysteresis seeds are allowed only in a lightly dilated seed ROI,
//     while weak continuation may travel in the broader ROI.
// ============================================================

constexpr float ROI_NORM_LOW_PERCENTILE  = 1.0f;
constexpr float ROI_NORM_HIGH_PERCENTILE = 99.7f;
constexpr float ROI_FALLBACK_BRIGHT_PERCENTILE = 86.0f;
constexpr int ROI_SAMPLE_STRIDE = 4;
constexpr int ROI_CLOSE_RADIUS = 4;
constexpr int ROI_SEED_DILATE_RADIUS = 6;
constexpr int ROI_DILATE_RADIUS = 28;
constexpr int ROI_MIN_COMPONENT_AREA = 200;
constexpr float ROI_DOMINANT_COMPONENT_RATIO = 0.08f;
constexpr float ROI_MIN_AREA_FRACTION = 0.0005f;
constexpr float ROI_MAX_AREA_FRACTION = 0.55f;
constexpr int ROI_Z_PROPAGATION_RADIUS = 1;

// ============================================================
// COMMON 3D NMS FOR BOTH CANNY VARIANTS
// ============================================================

constexpr float NMS_STEP = 1.0f;
constexpr float NMS_PLATEAU_FACTOR = 0.975f;

// ============================================================
// DETECTOR 1: GLOBAL / ORDINARY 3D CANNY
// ============================================================
// A single pair of thresholds is used for the whole 3D test window.
// The high threshold is selected from the global distribution of 3D-NMS
// gradient magnitudes; low is a fixed fraction of high.
// ============================================================

constexpr float STANDARD_HIGH_PERCENTILE = 90.0f;
constexpr float STANDARD_LOW_HIGH_RATIO = 0.40f;

// ============================================================
// DETECTOR 2: ADAPTIVE / NOISE-AWARE 3D CANNY
// ============================================================
// V2 follows the adaptive-Canny literature more closely:
//   * thresholds are derived from the gradient-magnitude histogram after NMS;
//   * Otsu determines the local high threshold automatically;
//   * low = 0.5 * high, matching a common Otsu-Canny construction;
//   * a GLOBAL Otsu reference is retained as a floor because purely local
//     thresholding is well known to collapse in quiet/noisy background tiles.
//
// The local and global thresholds are blended rather than switching abruptly.
// This is an engineering safeguard for XCT volumes with spatially varying
// reconstruction noise; it preserves the article's histogram-adaptive idea.
// ============================================================

constexpr int ADAPT_TILE_SIZE = 64;
constexpr int ADAPT_TILE_MARGIN = 24;
constexpr int ADAPT_Z_RADIUS = 2;
constexpr int ADAPT_SAMPLE_STRIDE = 2;
constexpr int ADAPT_MIN_SAMPLES = 40;
constexpr int ADAPT_OTSU_BINS = 256;
constexpr float ADAPT_LOW_HIGH_RATIO = 0.50f;
constexpr float ADAPT_GLOBAL_FLOOR_RATIO = 0.65f;
constexpr float ADAPT_GLOBAL_CEIL_RATIO  = 1.60f;
constexpr float ADAPT_LOCAL_BLEND_MIN = 0.35f;
constexpr float ADAPT_LOCAL_BLEND_MAX = 0.75f;
constexpr int ADAPT_GRID_SMOOTH_RADIUS_XY = 1;
constexpr int ADAPT_GRID_SMOOTH_RADIUS_Z = 1;

// ============================================================
// DETECTOR 3: NOISE-COMPENSATED PHASE CONGRUENCY
// ============================================================
// V2 implements the important parts of Kovesi's phasecong3 formulation more
// faithfully than V1:
//   * per-orientation phase congruency with Rayleigh noise compensation;
//   * frequency-spread weighting;
//   * covariance of phase congruency over orientations;
//   * MAXIMUM MOMENT M as the edge-strength measure;
//   * feature phase/type.  XCT material boundaries are predominantly step
//     features (feature phase near zero), so line/ridge-like responses receive
//     only a soft penalty instead of being discarded.
//
// Kovesi explicitly notes that k can be increased well above the default 2
// for noisy images.  We therefore use a more conservative value here while
// keeping the standard 4-scale/6-orientation Log-Gabor bank.
// ============================================================

constexpr int PC_NSCALE = 4;
constexpr int PC_NORIENT = 6;
constexpr float PC_MIN_WAVELENGTH = 3.0f;
constexpr float PC_MULT = 2.1f;
constexpr float PC_SIGMA_ON_F = 0.55f;
constexpr float PC_NOISE_K = 4.0f;
constexpr float PC_CUTOFF = 0.50f;
constexpr float PC_GAIN = 10.0f;
constexpr int PC_NOISE_SAMPLE_STRIDE = 4;
constexpr int PC_NOISE_BORDER = 8;
constexpr float PC_STEP_WEIGHT_FLOOR = 0.45f;
constexpr float PC_STEP_WEIGHT_POWER = 2.0f;
constexpr float PC_ANISOTROPY_WEIGHT_FLOOR = 0.35f;
constexpr float PC_NMS_STEP = 1.0f;
constexpr float PC_NMS_PLATEAU_FACTOR = 0.985f;
constexpr float PC_HIGH_PERCENTILE = 82.0f;
constexpr float PC_HIGH_FLOOR = 0.06f;
constexpr float PC_LOW_HIGH_RATIO = 0.40f;
constexpr float PC_LOW_FLOOR = 0.02f;

// ============================================================
// OUTPUT OPTIONS
// ============================================================

constexpr bool SAVE_ALL_SLICE_DIAGNOSTICS = true;
constexpr bool SAVE_PHASE_CONGRUENCY_16BIT = true;

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

struct RobustStats
{
    float median = 0.0f;
    float madSigma = 0.0f;
};

struct AdaptiveThresholdGrid
{
    int tilesX = 0;
    int tilesY = 0;
    int depth = 0;
    int tileSize = ADAPT_TILE_SIZE;

    std::vector<float> low;
    std::vector<float> high;

    inline size_t index(int tx, int ty, int z) const
    {
        return
            (static_cast<size_t>(z) * static_cast<size_t>(tilesY) +
             static_cast<size_t>(ty)) *
                static_cast<size_t>(tilesX) +
            static_cast<size_t>(tx);
    }

    inline int tileX(int x) const
    {
        return std::clamp(x / tileSize, 0, tilesX - 1);
    }

    inline int tileY(int y) const
    {
        return std::clamp(y / tileSize, 0, tilesY - 1);
    }

    inline float lowAt(int x, int y, int z) const
    {
        return low[index(tileX(x), tileY(y), z)];
    }

    inline float highAt(int x, int y, int z) const
    {
        return high[index(tileX(x), tileY(y), z)];
    }
};

struct PhaseFilterBank
{
    int originalRows = 0;
    int originalCols = 0;
    int paddedRows = 0;
    int paddedCols = 0;

    std::vector<cv::Mat> logGabor;
    std::vector<cv::Mat> spread;
};

struct PhaseCongruencyResult
{
    cv::Mat pc;          // CV_32F: Kovesi maximum moment M, edge strength
    cv::Mat stepness;    // CV_32F: 1 for step-like phase, 0 for line-like phase
    cv::Mat edgeScore;   // CV_32F: M with a soft step-feature weighting
    cv::Mat nmsScore;    // CV_32F: edgeScore only at local maxima
    float meanNoiseThreshold = 0.0f;
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

static RobustStats robustStats(std::vector<float> values)
{
    RobustStats out;

    if (values.empty())
        return out;

    out.median = percentile(values, 50.0f);

    for (float& v : values)
        v = std::abs(v - out.median);

    const float mad = percentile(values, 50.0f);
    out.madSigma = 1.4826f * mad;

    return out;
}


static float otsuThresholdFloatSamples(
    const std::vector<float>& values,
    int bins = 256)
{
    if (values.empty())
        return 0.0f;

    std::vector<float> finite;
    finite.reserve(values.size());

    for (float v : values)
    {
        if (std::isfinite(v) && v >= 0.0f)
            finite.push_back(v);
    }

    if (finite.empty())
        return 0.0f;

    const float hi = std::max(percentile(finite, 99.5f), 1e-12f);
    const float lo = 0.0f;

    if (hi <= lo + 1e-12f)
        return hi;

    bins = std::max(32, bins);
    std::vector<double> hist(static_cast<size_t>(bins), 0.0);

    for (float v : finite)
    {
        const float t = clamp01((v - lo) / (hi - lo));
        const int b = std::clamp(
            static_cast<int>(std::floor(t * static_cast<float>(bins - 1))),
            0,
            bins - 1
            );
        hist[static_cast<size_t>(b)] += 1.0;
    }

    double total = 0.0;
    double sumAll = 0.0;

    for (int i = 0; i < bins; ++i)
    {
        total += hist[static_cast<size_t>(i)];
        sumAll += static_cast<double>(i) * hist[static_cast<size_t>(i)];
    }

    if (total <= 0.0)
        return 0.0f;

    double weightBackground = 0.0;
    double sumBackground = 0.0;
    double bestVariance = -1.0;
    int bestBin = 0;

    for (int i = 0; i < bins; ++i)
    {
        const double h = hist[static_cast<size_t>(i)];
        weightBackground += h;
        sumBackground += static_cast<double>(i) * h;

        if (weightBackground <= 0.0)
            continue;

        const double weightForeground = total - weightBackground;
        if (weightForeground <= 0.0)
            break;

        const double meanBackground = sumBackground / weightBackground;
        const double meanForeground =
            (sumAll - sumBackground) / weightForeground;

        const double diff = meanBackground - meanForeground;
        const double between =
            weightBackground * weightForeground * diff * diff;

        if (between > bestVariance)
        {
            bestVariance = between;
            bestBin = i;
        }
    }

    const float fraction =
        static_cast<float>(bestBin) /
        static_cast<float>(std::max(1, bins - 1));

    return lo + fraction * (hi - lo);
}

static std::string sliceName(int globalZ)
{
    std::ostringstream ss;
    ss << std::setw(4) << std::setfill('0') << globalZ;
    return ss.str();
}

static size_t volumeIndex(
    int x,
    int y,
    int z,
    int width,
    int height)
{
    return
        (static_cast<size_t>(z) * static_cast<size_t>(height) +
         static_cast<size_t>(y)) *
            static_cast<size_t>(width) +
        static_cast<size_t>(x);
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
// OBJECT SUPPORT ROI: OTSU SEED + DOMINANT COMPONENTS
// ============================================================

static cv::Mat fillBinaryHoles(const cv::Mat& binary)
{
    cv::Mat padded;
    cv::copyMakeBorder(
        binary,
        padded,
        1, 1, 1, 1,
        cv::BORDER_CONSTANT,
        cv::Scalar(0)
        );

    cv::Mat flood = padded.clone();
    cv::floodFill(flood, cv::Point(0, 0), cv::Scalar(255));

    cv::Mat holes;
    cv::bitwise_not(flood, holes);

    cv::Mat filled;
    cv::bitwise_or(padded, holes, filled);

    return filled(
               cv::Rect(1, 1, binary.cols, binary.rows)
               ).clone();
}

static cv::Mat robustNormalize8U(const cv::Mat& input)
{
    CV_Assert(input.type() == CV_32F);

    std::vector<float> samples;
    samples.reserve(
        static_cast<size_t>((input.rows + ROI_SAMPLE_STRIDE - 1) /
                            ROI_SAMPLE_STRIDE) *
        static_cast<size_t>((input.cols + ROI_SAMPLE_STRIDE - 1) /
                            ROI_SAMPLE_STRIDE)
        );

    for (int y = 0; y < input.rows; y += ROI_SAMPLE_STRIDE)
    {
        const float* row = input.ptr<float>(y);
        for (int x = 0; x < input.cols; x += ROI_SAMPLE_STRIDE)
            samples.push_back(row[x]);
    }

    const float lo = percentile(samples, ROI_NORM_LOW_PERCENTILE);
    const float hi = percentile(samples, ROI_NORM_HIGH_PERCENTILE);
    const float range = std::max(hi - lo, 1e-8f);

    cv::Mat out(input.size(), CV_8U, cv::Scalar(0));

    for (int y = 0; y < input.rows; ++y)
    {
        const float* src = input.ptr<float>(y);
        uchar* dst = out.ptr<uchar>(y);

        for (int x = 0; x < input.cols; ++x)
        {
            dst[x] = cv::saturate_cast<uchar>(
                255.0f * clamp01((src[x] - lo) / range)
                );
        }
    }

    return out;
}

static cv::Mat keepDominantComponents(
    const cv::Mat& binary,
    int minArea,
    float relativeToLargest)
{
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;

    const int count = cv::connectedComponentsWithStats(
        binary,
        labels,
        stats,
        centroids,
        8,
        CV_32S
        );

    if (count <= 1)
        return cv::Mat(binary.size(), CV_8U, cv::Scalar(0));

    int largestArea = 0;
    for (int label = 1; label < count; ++label)
    {
        largestArea = std::max(
            largestArea,
            stats.at<int>(label, cv::CC_STAT_AREA)
            );
    }

    const int relativeFloor =
        static_cast<int>(std::lround(
            static_cast<double>(largestArea) *
            static_cast<double>(relativeToLargest)
            ));

    const int keepFloor = std::max(minArea, relativeFloor);

    cv::Mat out(binary.size(), CV_8U, cv::Scalar(0));

    for (int label = 1; label < count; ++label)
    {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < keepFloor)
            continue;

        out.setTo(255, labels == label);
    }

    return out;
}

static cv::Mat makeOtsuObjectSeed(const cv::Mat& slice)
{
    const cv::Mat normalized = robustNormalize8U(slice);

    cv::Mat seed;
    cv::threshold(
        normalized,
        seed,
        0.0,
        255.0,
        cv::THRESH_BINARY | cv::THRESH_OTSU
        );

    const cv::Mat closeKernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(2 * ROI_CLOSE_RADIUS + 1,
                 2 * ROI_CLOSE_RADIUS + 1)
        );

    cv::morphologyEx(
        seed,
        seed,
        cv::MORPH_CLOSE,
        closeKernel
        );

    seed = keepDominantComponents(
        seed,
        ROI_MIN_COMPONENT_AREA,
        ROI_DOMINANT_COMPONENT_RATIO
        );

    const double areaFraction =
        static_cast<double>(cv::countNonZero(seed)) /
        static_cast<double>(seed.total());

    // If Otsu degenerates because the reconstructed background dominates the
    // histogram, fall back to a conservative bright-tail seed.  The fallback
    // is still only a support mask and never becomes the measured boundary.
    if (areaFraction < ROI_MIN_AREA_FRACTION ||
        areaFraction > ROI_MAX_AREA_FRACTION)
    {
        std::vector<float> samples;
        samples.reserve(
            static_cast<size_t>((slice.rows + ROI_SAMPLE_STRIDE - 1) /
                                ROI_SAMPLE_STRIDE) *
            static_cast<size_t>((slice.cols + ROI_SAMPLE_STRIDE - 1) /
                                ROI_SAMPLE_STRIDE)
            );

        for (int y = 0; y < slice.rows; y += ROI_SAMPLE_STRIDE)
        {
            const float* row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; x += ROI_SAMPLE_STRIDE)
                samples.push_back(row[x]);
        }

        const float threshold =
            percentile(samples, ROI_FALLBACK_BRIGHT_PERCENTILE);

        cv::compare(slice, threshold, seed, cv::CMP_GE);
        cv::morphologyEx(seed, seed, cv::MORPH_CLOSE, closeKernel);
        seed = keepDominantComponents(
            seed,
            ROI_MIN_COMPONENT_AREA,
            ROI_DOMINANT_COMPONENT_RATIO
            );
    }

    return fillBinaryHoles(seed);
}

static void propagateMaskAlongZ(
    std::vector<uint8_t>& mask,
    int width,
    int height,
    int depth,
    int radius)
{
    if (radius <= 0)
        return;

    const size_t sliceSize =
        static_cast<size_t>(width) * static_cast<size_t>(height);

    const std::vector<uint8_t> original = mask;

    cv::parallel_for_(
        cv::Range(0, depth),
        [&](const cv::Range& range)
        {
            for (int z = range.start; z < range.end; ++z)
            {
                uint8_t* dst =
                    mask.data() + static_cast<size_t>(z) * sliceSize;

                for (int dz = -radius; dz <= radius; ++dz)
                {
                    const int zz = z + dz;
                    if (zz < 0 || zz >= depth)
                        continue;

                    const uint8_t* src =
                        original.data() + static_cast<size_t>(zz) * sliceSize;

                    for (size_t i = 0; i < sliceSize; ++i)
                    {
                        if (src[i] != 0)
                            dst[i] = 1;
                    }
                }
            }
        }
        );
}

static void buildROIVolumes(
    const std::vector<float>& volume,
    int width,
    int height,
    int depth,
    std::vector<uint8_t>& seedROI,
    std::vector<uint8_t>& broadROI)
{
    std::cout
        << "Building Otsu-based object support ROI..."
        << std::endl;

    const size_t sliceSize =
        static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t total =
        sliceSize * static_cast<size_t>(depth);

    seedROI.assign(total, 0);
    broadROI.assign(total, 0);

    const cv::Mat seedDilateKernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(2 * ROI_SEED_DILATE_RADIUS + 1,
                 2 * ROI_SEED_DILATE_RADIUS + 1)
        );

    const cv::Mat broadDilateKernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(2 * ROI_DILATE_RADIUS + 1,
                 2 * ROI_DILATE_RADIUS + 1)
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

        const cv::Mat filled = makeOtsuObjectSeed(slice);

        cv::Mat seedMask;
        cv::Mat broadMask;
        cv::dilate(filled, seedMask, seedDilateKernel);
        cv::dilate(filled, broadMask, broadDilateKernel);

        uint8_t* seedDst =
            seedROI.data() + static_cast<size_t>(z) * sliceSize;
        uint8_t* broadDst =
            broadROI.data() + static_cast<size_t>(z) * sliceSize;

        for (int y = 0; y < height; ++y)
        {
            const uchar* s = seedMask.ptr<uchar>(y);
            const uchar* b = broadMask.ptr<uchar>(y);

            for (int x = 0; x < width; ++x)
            {
                const size_t i = static_cast<size_t>(y) * width + x;
                seedDst[i] = s[x] != 0 ? 1 : 0;
                broadDst[i] = b[x] != 0 ? 1 : 0;
            }
        }
    }

    // Only the broad continuation mask is propagated through neighbouring
    // slices.  The seed mask remains tight so remote streaks cannot initiate
    // hysteresis merely because they occur near another slice's object.
    propagateMaskAlongZ(
        broadROI,
        width,
        height,
        depth,
        ROI_Z_PROPAGATION_RADIUS
        );
}

// ============================================================
// 3D GRADIENT
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
                                ? (volume[idxP] - volume[idxM]) /
                                      dzDistance
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
// TRUE 3D NMS FOR BOTH CANNY VARIANTS
// ============================================================

static std::vector<uint8_t> computeNMS3D(
    const GradientVolume& gradient)
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

    return nms;
}

static std::vector<float> collectNmsGradientSamples(
    const GradientVolume& gradient,
    const std::vector<uint8_t>& nms,
    const std::vector<uint8_t>& roi,
    int strideXY = 2)
{
    std::vector<float> samples;

    const size_t sliceSize =
        static_cast<size_t>(gradient.width) *
        static_cast<size_t>(gradient.height);

    for (int z = 1; z < gradient.depth - 1; ++z)
    {
        for (int y = 1; y < gradient.height - 1; y += strideXY)
        {
            for (int x = 1; x < gradient.width - 1; x += strideXY)
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

static std::vector<float> collectGradientNoiseSamples(
    const GradientVolume& gradient,
    const std::vector<uint8_t>& roi,
    int strideXY = 3)
{
    std::vector<float> samples;

    const size_t sliceSize =
        static_cast<size_t>(gradient.width) *
        static_cast<size_t>(gradient.height);

    for (int z = 0; z < gradient.depth; ++z)
    {
        for (int y = 0; y < gradient.height; y += strideXY)
        {
            for (int x = 0; x < gradient.width; x += strideXY)
            {
                const size_t idx =
                    static_cast<size_t>(z) * sliceSize +
                    static_cast<size_t>(y) * gradient.width +
                    static_cast<size_t>(x);

                if (roi[idx] != 0)
                    samples.push_back(gradient.data[idx].mag);
            }
        }
    }

    return samples;
}

// ============================================================
// DETECTOR 1: GLOBAL 3D CANNY HYSTERESIS
// ============================================================

static std::vector<uint8_t> hysteresisGlobalGradient(
    const GradientVolume& gradient,
    const std::vector<uint8_t>& nms,
    const std::vector<uint8_t>& seedROI,
    const std::vector<uint8_t>& supportROI,
    float lowThreshold,
    float highThreshold)
{
    std::cout << "Running ordinary/global 3D Canny hysteresis..."
              << std::endl;

    const int width = gradient.width;
    const int height = gradient.height;
    const int depth = gradient.depth;

    const size_t total = gradient.data.size();
    const size_t sliceSize =
        static_cast<size_t>(width) * static_cast<size_t>(height);

    std::vector<uint8_t> accepted(total, 0);
    std::vector<uint32_t> queue;
    queue.reserve(total / 50 + 1024);

    // Strong edges may seed only near the object.  This is the key defence
    // against unrelated strong reconstruction streaks far from the part.
    for (size_t idx = 0; idx < total; ++idx)
    {
        if (nms[idx] == 0 || seedROI[idx] == 0)
            continue;

        if (gradient.data[idx].mag >= highThreshold)
        {
            accepted[idx] = 1;
            queue.push_back(static_cast<uint32_t>(idx));
        }
    }

    size_t head = 0;

    while (head < queue.size())
    {
        const size_t idx = queue[head++];

        const int z = static_cast<int>(idx / sliceSize);
        const size_t rem = idx - static_cast<size_t>(z) * sliceSize;
        const int y = static_cast<int>(rem / static_cast<size_t>(width));
        const int x = static_cast<int>(rem % static_cast<size_t>(width));

        for (int dz = -1; dz <= 1; ++dz)
        {
            const int zz = z + dz;
            if (zz < 0 || zz >= depth)
                continue;

            for (int dy = -1; dy <= 1; ++dy)
            {
                const int yy = y + dy;
                if (yy < 0 || yy >= height)
                    continue;

                for (int dx = -1; dx <= 1; ++dx)
                {
                    if (dx == 0 && dy == 0 && dz == 0)
                        continue;

                    const int xx = x + dx;
                    if (xx < 0 || xx >= width)
                        continue;

                    const size_t nidx =
                        static_cast<size_t>(zz) * sliceSize +
                        static_cast<size_t>(yy) * width +
                        static_cast<size_t>(xx);

                    if (accepted[nidx] != 0 ||
                        nms[nidx] == 0 ||
                        supportROI[nidx] == 0)
                    {
                        continue;
                    }

                    if (gradient.data[nidx].mag >= lowThreshold)
                    {
                        accepted[nidx] = 1;
                        queue.push_back(static_cast<uint32_t>(nidx));
                    }
                }
            }
        }
    }

    return accepted;
}

// ============================================================
// DETECTOR 2: ADAPTIVE OTSU THRESHOLD GRID
// ============================================================

static AdaptiveThresholdGrid buildAdaptiveThresholdGrid(
    const GradientVolume& gradient,
    const std::vector<uint8_t>& nms,
    const std::vector<uint8_t>& supportROI,
    float globalHigh)
{
    std::cout << "Building adaptive Otsu threshold field..."
              << std::endl;

    AdaptiveThresholdGrid grid;
    grid.tilesX =
        (gradient.width + ADAPT_TILE_SIZE - 1) / ADAPT_TILE_SIZE;
    grid.tilesY =
        (gradient.height + ADAPT_TILE_SIZE - 1) / ADAPT_TILE_SIZE;
    grid.depth = gradient.depth;

    const size_t gridSize =
        static_cast<size_t>(grid.tilesX) *
        static_cast<size_t>(grid.tilesY) *
        static_cast<size_t>(grid.depth);

    grid.low.assign(gridSize, 0.0f);
    grid.high.assign(gridSize, 0.0f);

    const size_t sliceSize =
        static_cast<size_t>(gradient.width) *
        static_cast<size_t>(gradient.height);

    const float safeGlobalHigh = std::max(globalHigh, 1e-12f);

    for (int z = 0; z < grid.depth; ++z)
    {
        std::cout << "\rAdaptive Otsu statistics: slice "
                  << (FIRST_SLICE + z)
                  << " / " << LAST_SLICE
                  << std::flush;

        for (int ty = 0; ty < grid.tilesY; ++ty)
        {
            for (int tx = 0; tx < grid.tilesX; ++tx)
            {
                const int baseX0 = tx * ADAPT_TILE_SIZE;
                const int baseY0 = ty * ADAPT_TILE_SIZE;
                const int baseX1 =
                    std::min(gradient.width,
                             (tx + 1) * ADAPT_TILE_SIZE);
                const int baseY1 =
                    std::min(gradient.height,
                             (ty + 1) * ADAPT_TILE_SIZE);

                const int x0 =
                    std::max(1, baseX0 - ADAPT_TILE_MARGIN);
                const int y0 =
                    std::max(1, baseY0 - ADAPT_TILE_MARGIN);
                const int x1 =
                    std::min(gradient.width - 1,
                             baseX1 + ADAPT_TILE_MARGIN);
                const int y1 =
                    std::min(gradient.height - 1,
                             baseY1 + ADAPT_TILE_MARGIN);

                const int z0 = std::max(0, z - ADAPT_Z_RADIUS);
                const int z1 = std::min(grid.depth - 1,
                                        z + ADAPT_Z_RADIUS);

                std::vector<float> localNms;
                localNms.reserve(2048);

                for (int zz = z0; zz <= z1; ++zz)
                {
                    for (int y = y0; y < y1; y += ADAPT_SAMPLE_STRIDE)
                    {
                        for (int x = x0; x < x1; x += ADAPT_SAMPLE_STRIDE)
                        {
                            const size_t idx =
                                static_cast<size_t>(zz) * sliceSize +
                                static_cast<size_t>(y) * gradient.width +
                                static_cast<size_t>(x);

                            if (supportROI[idx] == 0 || nms[idx] == 0)
                                continue;

                            const float g = gradient.data[idx].mag;
                            if (std::isfinite(g) && g > 0.0f)
                                localNms.push_back(g);
                        }
                    }
                }

                float localHigh = safeGlobalHigh;

                if (static_cast<int>(localNms.size()) >= ADAPT_MIN_SAMPLES)
                {
                    localHigh = otsuThresholdFloatSamples(
                        localNms,
                        ADAPT_OTSU_BINS
                        );

                    if (!std::isfinite(localHigh) || localHigh <= 1e-12f)
                        localHigh = safeGlobalHigh;
                }

                // More local samples -> trust the local histogram more.
                const float sampleReliability = clamp01(
                    static_cast<float>(localNms.size()) /
                    static_cast<float>(std::max(1, ADAPT_MIN_SAMPLES * 8))
                    );

                const float blend =
                    ADAPT_LOCAL_BLEND_MIN +
                    (ADAPT_LOCAL_BLEND_MAX - ADAPT_LOCAL_BLEND_MIN) *
                        sampleReliability;

                float high =
                    (1.0f - blend) * safeGlobalHigh +
                    blend * localHigh;

                // Literature-driven adaptive thresholding, but with a global
                // floor to prevent near-zero thresholds in quiet background.
                high = std::clamp(
                    high,
                    safeGlobalHigh * ADAPT_GLOBAL_FLOOR_RATIO,
                    safeGlobalHigh * ADAPT_GLOBAL_CEIL_RATIO
                    );

                const float low =
                    high * ADAPT_LOW_HIGH_RATIO;

                const size_t gidx = grid.index(tx, ty, z);
                grid.low[gidx] = low;
                grid.high[gidx] = high;
            }
        }
    }

    std::cout << std::endl;

    // Smooth the threshold grid to avoid block seams.  This does not alter
    // the underlying gradient/NMS; it only regularises local threshold choice.
    std::vector<float> lowSmoothed(grid.low.size(), 0.0f);
    std::vector<float> highSmoothed(grid.high.size(), 0.0f);

    for (int z = 0; z < grid.depth; ++z)
    {
        for (int ty = 0; ty < grid.tilesY; ++ty)
        {
            for (int tx = 0; tx < grid.tilesX; ++tx)
            {
                double sumLow = 0.0;
                double sumHigh = 0.0;
                int count = 0;

                for (int dz = -ADAPT_GRID_SMOOTH_RADIUS_Z;
                     dz <= ADAPT_GRID_SMOOTH_RADIUS_Z;
                     ++dz)
                {
                    const int zz = z + dz;
                    if (zz < 0 || zz >= grid.depth)
                        continue;

                    for (int dy = -ADAPT_GRID_SMOOTH_RADIUS_XY;
                         dy <= ADAPT_GRID_SMOOTH_RADIUS_XY;
                         ++dy)
                    {
                        const int yy = ty + dy;
                        if (yy < 0 || yy >= grid.tilesY)
                            continue;

                        for (int dx = -ADAPT_GRID_SMOOTH_RADIUS_XY;
                             dx <= ADAPT_GRID_SMOOTH_RADIUS_XY;
                             ++dx)
                        {
                            const int xx = tx + dx;
                            if (xx < 0 || xx >= grid.tilesX)
                                continue;

                            const size_t idx = grid.index(xx, yy, zz);
                            sumLow += grid.low[idx];
                            sumHigh += grid.high[idx];
                            ++count;
                        }
                    }
                }

                const size_t idx = grid.index(tx, ty, z);
                lowSmoothed[idx] =
                    static_cast<float>(sumLow / std::max(count, 1));
                highSmoothed[idx] =
                    static_cast<float>(sumHigh / std::max(count, 1));
            }
        }
    }

    grid.low.swap(lowSmoothed);
    grid.high.swap(highSmoothed);

    return grid;
}

static std::vector<uint8_t> hysteresisAdaptiveGradient(
    const GradientVolume& gradient,
    const std::vector<uint8_t>& nms,
    const std::vector<uint8_t>& seedROI,
    const std::vector<uint8_t>& supportROI,
    const AdaptiveThresholdGrid& grid)
{
    std::cout << "Running adaptive/Otsu 3D Canny hysteresis..."
              << std::endl;

    const int width = gradient.width;
    const int height = gradient.height;
    const int depth = gradient.depth;

    const size_t total = gradient.data.size();
    const size_t sliceSize =
        static_cast<size_t>(width) * static_cast<size_t>(height);

    std::vector<uint8_t> accepted(total, 0);
    std::vector<uint32_t> queue;
    queue.reserve(total / 50 + 1024);

    for (int z = 0; z < depth; ++z)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const size_t idx =
                    static_cast<size_t>(z) * sliceSize +
                    static_cast<size_t>(y) * width +
                    static_cast<size_t>(x);

                if (nms[idx] == 0 || seedROI[idx] == 0)
                    continue;

                if (gradient.data[idx].mag >= grid.highAt(x, y, z))
                {
                    accepted[idx] = 1;
                    queue.push_back(static_cast<uint32_t>(idx));
                }
            }
        }
    }

    size_t head = 0;

    while (head < queue.size())
    {
        const size_t idx = queue[head++];

        const int z = static_cast<int>(idx / sliceSize);
        const size_t rem = idx - static_cast<size_t>(z) * sliceSize;
        const int y = static_cast<int>(rem / static_cast<size_t>(width));
        const int x = static_cast<int>(rem % static_cast<size_t>(width));

        for (int dz = -1; dz <= 1; ++dz)
        {
            const int zz = z + dz;
            if (zz < 0 || zz >= depth)
                continue;

            for (int dy = -1; dy <= 1; ++dy)
            {
                const int yy = y + dy;
                if (yy < 0 || yy >= height)
                    continue;

                for (int dx = -1; dx <= 1; ++dx)
                {
                    if (dx == 0 && dy == 0 && dz == 0)
                        continue;

                    const int xx = x + dx;
                    if (xx < 0 || xx >= width)
                        continue;

                    const size_t nidx =
                        static_cast<size_t>(zz) * sliceSize +
                        static_cast<size_t>(yy) * width +
                        static_cast<size_t>(xx);

                    if (accepted[nidx] != 0 ||
                        nms[nidx] == 0 ||
                        supportROI[nidx] == 0)
                    {
                        continue;
                    }

                    if (gradient.data[nidx].mag >=
                        grid.lowAt(xx, yy, zz))
                    {
                        accepted[nidx] = 1;
                        queue.push_back(static_cast<uint32_t>(nidx));
                    }
                }
            }
        }
    }

    return accepted;
}

// ============================================================
// PHASE CONGRUENCY FILTER BANK
// ============================================================

static PhaseFilterBank buildPhaseFilterBank(int rows, int cols)
{
    std::cout << "Building Log-Gabor phase-congruency filter bank..."
              << std::endl;

    PhaseFilterBank bank;
    bank.originalRows = rows;
    bank.originalCols = cols;
    bank.paddedRows = cv::getOptimalDFTSize(rows);
    bank.paddedCols = cv::getOptimalDFTSize(cols);

    cv::Mat radius(bank.paddedRows, bank.paddedCols, CV_32F);
    cv::Mat theta(bank.paddedRows, bank.paddedCols, CV_32F);
    cv::Mat lowpass(bank.paddedRows, bank.paddedCols, CV_32F);

    constexpr float PI = 3.14159265358979323846f;

    for (int y = 0; y < bank.paddedRows; ++y)
    {
        float* rRow = radius.ptr<float>(y);
        float* tRow = theta.ptr<float>(y);
        float* lRow = lowpass.ptr<float>(y);

        const float fy =
            (y <= bank.paddedRows / 2)
                ? static_cast<float>(y) /
                      static_cast<float>(bank.paddedRows)
                : static_cast<float>(y - bank.paddedRows) /
                      static_cast<float>(bank.paddedRows);

        for (int x = 0; x < bank.paddedCols; ++x)
        {
            const float fx =
                (x <= bank.paddedCols / 2)
                    ? static_cast<float>(x) /
                          static_cast<float>(bank.paddedCols)
                    : static_cast<float>(x - bank.paddedCols) /
                          static_cast<float>(bank.paddedCols);

            const float r = std::sqrt(fx * fx + fy * fy);
            rRow[x] = r;
            tRow[x] = std::atan2(-fy, fx);

            // Butterworth-like low-pass used only to tame the DFT corners.
            const float ratio = r / 0.45f;
            lRow[x] =
                1.0f /
                (1.0f + std::pow(std::max(ratio, 0.0f), 30.0f));
        }
    }

    radius.at<float>(0, 0) = 1.0f;

    bank.logGabor.resize(PC_NSCALE);

    const float logSigma = std::log(PC_SIGMA_ON_F);
    const float denom = 2.0f * logSigma * logSigma;

    for (int s = 0; s < PC_NSCALE; ++s)
    {
        const float wavelength =
            PC_MIN_WAVELENGTH *
            std::pow(PC_MULT, static_cast<float>(s));

        const float fo = 1.0f / wavelength;

        cv::Mat lg(bank.paddedRows, bank.paddedCols, CV_32F);

        for (int y = 0; y < bank.paddedRows; ++y)
        {
            const float* rRow = radius.ptr<float>(y);
            const float* lRow = lowpass.ptr<float>(y);
            float* gRow = lg.ptr<float>(y);

            for (int x = 0; x < bank.paddedCols; ++x)
            {
                const float lr = std::log(rRow[x] / fo);
                gRow[x] =
                    std::exp(-(lr * lr) / denom) * lRow[x];
            }
        }

        lg.at<float>(0, 0) = 0.0f;
        bank.logGabor[s] = std::move(lg);
    }

    bank.spread.resize(PC_NORIENT);

    for (int o = 0; o < PC_NORIENT; ++o)
    {
        const float angle =
            static_cast<float>(o) * PI /
            static_cast<float>(PC_NORIENT);

        const float sinA = std::sin(angle);
        const float cosA = std::cos(angle);

        cv::Mat spread(bank.paddedRows, bank.paddedCols, CV_32F);

        for (int y = 0; y < bank.paddedRows; ++y)
        {
            const float* tRow = theta.ptr<float>(y);
            float* sRow = spread.ptr<float>(y);

            for (int x = 0; x < bank.paddedCols; ++x)
            {
                const float th = tRow[x];
                const float sinT = std::sin(th);
                const float cosT = std::cos(th);

                const float ds = sinT * cosA - cosT * sinA;
                const float dc = cosT * cosA + sinT * sinA;

                float dtheta = std::abs(std::atan2(ds, dc));
                dtheta =
                    std::min(
                        dtheta *
                            static_cast<float>(PC_NORIENT) / 2.0f,
                        PI
                        );

                sRow[x] = 0.5f * (std::cos(dtheta) + 1.0f);
            }
        }

        bank.spread[o] = std::move(spread);
    }

    return bank;
}

static float sampleMatBilinear(
    const cv::Mat& image,
    float x,
    float y)
{
    if (x < 0.0f || y < 0.0f ||
        x >= static_cast<float>(image.cols - 1) ||
        y >= static_cast<float>(image.rows - 1))
    {
        return 0.0f;
    }

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);

    const float v00 = image.at<float>(y0, x0);
    const float v10 = image.at<float>(y0, x1);
    const float v01 = image.at<float>(y1, x0);
    const float v11 = image.at<float>(y1, x1);

    const float v0 = v00 * (1.0f - tx) + v10 * tx;
    const float v1 = v01 * (1.0f - tx) + v11 * tx;

    return v0 * (1.0f - ty) + v1 * ty;
}

static float estimateRayleighTau(const cv::Mat& amplitude)
{
    std::vector<float> samples;

    const int border = std::min(
        PC_NOISE_BORDER,
        std::max(0, std::min(amplitude.rows, amplitude.cols) / 8)
        );

    for (int y = border;
         y < amplitude.rows - border;
         y += PC_NOISE_SAMPLE_STRIDE)
    {
        const float* a = amplitude.ptr<float>(y);

        for (int x = border;
             x < amplitude.cols - border;
             x += PC_NOISE_SAMPLE_STRIDE)
        {
            const float v = a[x];
            if (std::isfinite(v) && v >= 0.0f)
                samples.push_back(v);
        }
    }

    if (samples.empty())
        return 0.0f;

    const float med = percentile(samples, 50.0f);

    // Kovesi's noiseMethod = -1:
    // median(Rayleigh) = tau * sqrt(log(4)).
    return med / std::sqrt(std::log(4.0f));
}

static PhaseCongruencyResult computePhaseCongruency2D(
    const cv::Mat& input,
    const cv::Mat& roiMask,
    const PhaseFilterBank& bank)
{
    CV_Assert(input.type() == CV_32F);
    CV_Assert(roiMask.type() == CV_8U);

    cv::Mat padded;
    cv::copyMakeBorder(
        input,
        padded,
        0,
        bank.paddedRows - input.rows,
        0,
        bank.paddedCols - input.cols,
        cv::BORDER_REFLECT_101
        );

    cv::Mat spectrum;
    cv::dft(padded, spectrum, cv::DFT_COMPLEX_OUTPUT);

    std::vector<cv::Mat> spectrumPlanes;
    cv::split(spectrum, spectrumPlanes);

    cv::Mat covx2(input.size(), CV_32F, cv::Scalar(0));
    cv::Mat covy2(input.size(), CV_32F, cv::Scalar(0));
    cv::Mat covxy(input.size(), CV_32F, cv::Scalar(0));

    cv::Mat energyEven(input.size(), CV_32F, cv::Scalar(0));
    cv::Mat energyOddX(input.size(), CV_32F, cv::Scalar(0));
    cv::Mat energyOddY(input.size(), CV_32F, cv::Scalar(0));

    float noiseThresholdSum = 0.0f;

    constexpr float PI = 3.14159265358979323846f;
    const cv::Rect crop(0, 0, input.cols, input.rows);

    for (int o = 0; o < PC_NORIENT; ++o)
    {
        const float theta =
            static_cast<float>(o) * PI /
            static_cast<float>(PC_NORIENT);

        cv::Mat sumE(input.size(), CV_32F, cv::Scalar(0));
        cv::Mat sumO(input.size(), CV_32F, cv::Scalar(0));
        cv::Mat sumAn(input.size(), CV_32F, cv::Scalar(0));
        cv::Mat maxAn(input.size(), CV_32F, cv::Scalar(0));

        std::vector<cv::Mat> evenResponses;
        std::vector<cv::Mat> oddResponses;
        evenResponses.reserve(PC_NSCALE);
        oddResponses.reserve(PC_NSCALE);

        float tau = 0.0f;

        for (int scale = 0; scale < PC_NSCALE; ++scale)
        {
            const cv::Mat filter =
                bank.logGabor[scale].mul(bank.spread[o]);

            cv::Mat filteredReal = spectrumPlanes[0].mul(filter);
            cv::Mat filteredImag = spectrumPlanes[1].mul(filter);

            std::vector<cv::Mat> filteredPlanes =
                {filteredReal, filteredImag};

            cv::Mat filteredSpectrum;
            cv::merge(filteredPlanes, filteredSpectrum);

            cv::Mat spatialComplex;
            cv::dft(
                filteredSpectrum,
                spatialComplex,
                cv::DFT_INVERSE |
                    cv::DFT_SCALE |
                    cv::DFT_COMPLEX_OUTPUT
                );

            std::vector<cv::Mat> responsePlanes;
            cv::split(spatialComplex, responsePlanes);

            cv::Mat even = responsePlanes[0](crop).clone();
            cv::Mat odd  = responsePlanes[1](crop).clone();

            cv::Mat amplitude;
            cv::magnitude(even, odd, amplitude);

            sumE += even;
            sumO += odd;
            sumAn += amplitude;
            cv::max(maxAn, amplitude, maxAn);

            if (scale == 0)
                tau = estimateRayleighTau(amplitude);

            evenResponses.push_back(std::move(even));
            oddResponses.push_back(std::move(odd));
        }

        // Kovesi phasecong3: accumulate overall energy vector for feature
        // orientation and feature type/phase.
        energyEven += sumE;
        energyOddX += sumO * std::cos(theta);
        energyOddY += sumO * std::sin(theta);

        cv::Mat xEnergy;
        cv::magnitude(sumE, sumO, xEnergy);

        cv::Mat meanE(input.size(), CV_32F);
        cv::Mat meanO(input.size(), CV_32F);

        for (int y = 0; y < input.rows; ++y)
        {
            const float* se = sumE.ptr<float>(y);
            const float* so = sumO.ptr<float>(y);
            const float* xe = xEnergy.ptr<float>(y);
            float* me = meanE.ptr<float>(y);
            float* mo = meanO.ptr<float>(y);

            for (int x = 0; x < input.cols; ++x)
            {
                const float denom = xe[x] + 1e-12f;
                me[x] = se[x] / denom;
                mo[x] = so[x] / denom;
            }
        }

        cv::Mat energy(input.size(), CV_32F, cv::Scalar(0));

        for (int scale = 0; scale < PC_NSCALE; ++scale)
        {
            const cv::Mat& even = evenResponses[scale];
            const cv::Mat& odd  = oddResponses[scale];

            for (int y = 0; y < input.rows; ++y)
            {
                const float* er = even.ptr<float>(y);
                const float* orow = odd.ptr<float>(y);
                const float* mer = meanE.ptr<float>(y);
                const float* mor = meanO.ptr<float>(y);
                float* en = energy.ptr<float>(y);

                for (int x = 0; x < input.cols; ++x)
                {
                    const float dot =
                        er[x] * mer[x] + orow[x] * mor[x];

                    const float cross = std::abs(
                        er[x] * mor[x] - orow[x] * mer[x]
                        );

                    en[x] += dot - cross;
                }
            }
        }

        // Noise compensation from Kovesi: estimate the Rayleigh parameter
        // from the smallest-scale amplitude and extrapolate through scales.
        const float invMult = 1.0f / PC_MULT;
        float scaleSum = 0.0f;
        float power = 1.0f;

        for (int scale = 0; scale < PC_NSCALE; ++scale)
        {
            scaleSum += power;
            power *= invMult;
        }

        const float totalTau = tau * scaleSum;
        const float noiseMean =
            totalTau * std::sqrt(PI / 2.0f);
        const float noiseSigma =
            totalTau * std::sqrt((4.0f - PI) / 2.0f);
        const float noiseThreshold = std::max(
            noiseMean + PC_NOISE_K * noiseSigma,
            0.0f
            );

        noiseThresholdSum += noiseThreshold;

        cv::Mat pcOrientation(input.size(), CV_32F, cv::Scalar(0));

        for (int y = 0; y < input.rows; ++y)
        {
            const float* en = energy.ptr<float>(y);
            const float* sa = sumAn.ptr<float>(y);
            const float* ma = maxAn.ptr<float>(y);
            float* pc = pcOrientation.ptr<float>(y);

            for (int x = 0; x < input.cols; ++x)
            {
                const float corrected =
                    std::max(en[x] - noiseThreshold, 0.0f);

                if (corrected <= 0.0f || sa[x] <= 1e-12f)
                    continue;

                float width = 0.0f;
                if (PC_NSCALE > 1 && ma[x] > 1e-12f)
                {
                    width =
                        (sa[x] / (ma[x] + 1e-12f) - 1.0f) /
                        static_cast<float>(PC_NSCALE - 1);
                }

                width = clamp01(width);

                const float frequencyWeight =
                    1.0f /
                    (1.0f +
                     std::exp((PC_CUTOFF - width) * PC_GAIN));

                pc[x] = clamp01(
                    frequencyWeight * corrected /
                    (sa[x] + 1e-12f)
                    );
            }
        }

        // Kovesi 2003: build covariance of phase congruency over orientation.
        const float c = std::cos(theta);
        const float sn = std::sin(theta);

        for (int y = 0; y < input.rows; ++y)
        {
            const float* p = pcOrientation.ptr<float>(y);
            float* xx = covx2.ptr<float>(y);
            float* yy = covy2.ptr<float>(y);
            float* xy = covxy.ptr<float>(y);

            for (int x = 0; x < input.cols; ++x)
            {
                const float covx = p[x] * c;
                const float covy = p[x] * sn;

                xx[x] += covx * covx;
                yy[x] += covy * covy;
                xy[x] += covx * covy;
            }
        }
    }

    const float orientNorm =
        static_cast<float>(PC_NORIENT) / 2.0f;

    cv::Mat maximumMoment(input.size(), CV_32F, cv::Scalar(0));
    cv::Mat minimumMoment(input.size(), CV_32F, cv::Scalar(0));
    cv::Mat orientation(input.size(), CV_32F, cv::Scalar(0));
    cv::Mat stepness(input.size(), CV_32F, cv::Scalar(0));
    cv::Mat edgeScore(input.size(), CV_32F, cv::Scalar(0));

    for (int y = 0; y < input.rows; ++y)
    {
        const float* cxx = covx2.ptr<float>(y);
        const float* cyy = covy2.ptr<float>(y);
        const float* cxy = covxy.ptr<float>(y);

        const float* ev0 = energyEven.ptr<float>(y);
        const float* evx = energyOddX.ptr<float>(y);
        const float* evy = energyOddY.ptr<float>(y);
        const uchar* roi = roiMask.ptr<uchar>(y);

        float* M = maximumMoment.ptr<float>(y);
        float* m = minimumMoment.ptr<float>(y);
        float* ori = orientation.ptr<float>(y);
        float* st = stepness.ptr<float>(y);
        float* es = edgeScore.ptr<float>(y);

        for (int x = 0; x < input.cols; ++x)
        {
            if (roi[x] == 0)
                continue;

            const float xx = cxx[x] / orientNorm;
            const float yy = cyy[x] / orientNorm;
            const float xy = 4.0f * cxy[x] /
                             static_cast<float>(PC_NORIENT);

            const float denom = std::sqrt(
                                    xy * xy + (xx - yy) * (xx - yy)
                                    ) + 1e-12f;

            const float maxMoment =
                0.5f * (xx + yy + denom);
            const float minMoment =
                0.5f * (xx + yy - denom);

            M[x] = clamp01(maxMoment);
            m[x] = clamp01(std::max(minMoment, 0.0f));

            float theta = std::atan2(evy[x], evx[x]);
            if (theta < 0.0f)
                theta += PI;
            ori[x] = theta;

            const float oddMagnitude =
                std::sqrt(evx[x] * evx[x] + evy[x] * evy[x]);

            // Kovesi feature type: 0 = step, +/-pi/2 = bright/dark line.
            const float featureType =
                std::atan2(ev0[x], oddMagnitude + 1e-12f);

            const float step = std::pow(
                std::max(0.0f, std::abs(std::cos(featureType))),
                PC_STEP_WEIGHT_POWER
                );

            st[x] = clamp01(step);

            // Edge-like orientation structure has M >> m.  If M and m are
            // both high the point is more corner/texture-like.  This weighting
            // uses the same moments introduced by Kovesi rather than adding a
            // new gradient-amplitude criterion.
            const float anisotropy = clamp01(
                (M[x] - m[x]) / (M[x] + 1e-12f)
                );

            const float stepWeight =
                PC_STEP_WEIGHT_FLOOR +
                (1.0f - PC_STEP_WEIGHT_FLOOR) * st[x];

            const float anisotropyWeight =
                PC_ANISOTROPY_WEIGHT_FLOOR +
                (1.0f - PC_ANISOTROPY_WEIGHT_FLOOR) * anisotropy;

            es[x] = clamp01(M[x] * stepWeight * anisotropyWeight);
        }
    }

    cv::Mat nmsScore(input.size(), CV_32F, cv::Scalar(0));

    for (int y = 1; y < input.rows - 1; ++y)
    {
        const float* scoreRow = edgeScore.ptr<float>(y);
        const float* thetaRow = orientation.ptr<float>(y);
        const uchar* roiRow = roiMask.ptr<uchar>(y);
        float* out = nmsScore.ptr<float>(y);

        for (int x = 1; x < input.cols - 1; ++x)
        {
            if (roiRow[x] == 0 || scoreRow[x] <= 0.0f)
                continue;

            // Kovesi orientation: 0 corresponds to a vertical edge, so the
            // normal direction used by NMS is (cos(theta), -sin(theta)).
            const float theta = thetaRow[x];
            const float dx = std::cos(theta) * PC_NMS_STEP;
            const float dy = -std::sin(theta) * PC_NMS_STEP;

            const float plus = sampleMatBilinear(
                edgeScore,
                static_cast<float>(x) + dx,
                static_cast<float>(y) + dy
                );

            const float minus = sampleMatBilinear(
                edgeScore,
                static_cast<float>(x) - dx,
                static_cast<float>(y) - dy
                );

            if (scoreRow[x] >= plus * PC_NMS_PLATEAU_FACTOR &&
                scoreRow[x] >= minus * PC_NMS_PLATEAU_FACTOR)
            {
                out[x] = scoreRow[x];
            }
        }
    }

    PhaseCongruencyResult result;
    result.pc = std::move(maximumMoment);
    result.stepness = std::move(stepness);
    result.edgeScore = std::move(edgeScore);
    result.nmsScore = std::move(nmsScore);
    result.meanNoiseThreshold =
        noiseThresholdSum / static_cast<float>(PC_NORIENT);

    return result;
}

// ============================================================
// GENERIC 3D HYSTERESIS FOR PHASE-CONGRUENCY SCORE
// ============================================================

static std::vector<uint8_t> hysteresisFloatScore3D(
    const std::vector<float>& score,
    const std::vector<uint8_t>& seedROI,
    const std::vector<uint8_t>& supportROI,
    int width,
    int height,
    int depth,
    float lowThreshold,
    float highThreshold)
{
    const size_t total = score.size();
    const size_t sliceSize =
        static_cast<size_t>(width) * static_cast<size_t>(height);

    std::vector<uint8_t> accepted(total, 0);
    std::vector<uint32_t> queue;
    queue.reserve(total / 50 + 1024);

    for (size_t idx = 0; idx < total; ++idx)
    {
        if (seedROI[idx] == 0 || score[idx] <= 0.0f)
            continue;

        if (score[idx] >= highThreshold)
        {
            accepted[idx] = 1;
            queue.push_back(static_cast<uint32_t>(idx));
        }
    }

    size_t head = 0;

    while (head < queue.size())
    {
        const size_t idx = queue[head++];

        const int z = static_cast<int>(idx / sliceSize);
        const size_t rem = idx - static_cast<size_t>(z) * sliceSize;
        const int y = static_cast<int>(rem / static_cast<size_t>(width));
        const int x = static_cast<int>(rem % static_cast<size_t>(width));

        for (int dz = -1; dz <= 1; ++dz)
        {
            const int zz = z + dz;
            if (zz < 0 || zz >= depth)
                continue;

            for (int dy = -1; dy <= 1; ++dy)
            {
                const int yy = y + dy;
                if (yy < 0 || yy >= height)
                    continue;

                for (int dx = -1; dx <= 1; ++dx)
                {
                    if (dx == 0 && dy == 0 && dz == 0)
                        continue;

                    const int xx = x + dx;
                    if (xx < 0 || xx >= width)
                        continue;

                    const size_t nidx =
                        static_cast<size_t>(zz) * sliceSize +
                        static_cast<size_t>(yy) * width +
                        static_cast<size_t>(xx);

                    if (accepted[nidx] != 0 ||
                        supportROI[nidx] == 0 ||
                        score[nidx] < lowThreshold)
                    {
                        continue;
                    }

                    accepted[nidx] = 1;
                    queue.push_back(static_cast<uint32_t>(nidx));
                }
            }
        }
    }

    return accepted;
}

// ============================================================
// DIAGNOSTIC IMAGE HELPERS
// ============================================================

static cv::Mat maskSliceFromVolume(
    const std::vector<uint8_t>& volume,
    int width,
    int height,
    int z)
{
    cv::Mat mask(height, width, CV_8U, cv::Scalar(0));

    const size_t sliceSize =
        static_cast<size_t>(width) * static_cast<size_t>(height);

    const uint8_t* src =
        volume.data() +
        static_cast<size_t>(z) * sliceSize;

    for (int y = 0; y < height; ++y)
    {
        uchar* dst = mask.ptr<uchar>(y);

        for (int x = 0; x < width; ++x)
        {
            dst[x] =
                src[static_cast<size_t>(y) * width + x] != 0
                    ? 255
                    : 0;
        }
    }

    return mask;
}

static cv::Mat gradientSlicePreview(
    const GradientVolume& gradient,
    const std::vector<uint8_t>& roi,
    int z)
{
    const int width = gradient.width;
    const int height = gradient.height;
    const size_t sliceSize =
        static_cast<size_t>(width) * static_cast<size_t>(height);

    std::vector<float> samples;

    for (int y = 0; y < height; y += 2)
    {
        for (int x = 0; x < width; x += 2)
        {
            const size_t idx =
                static_cast<size_t>(z) * sliceSize +
                static_cast<size_t>(y) * width +
                static_cast<size_t>(x);

            if (roi[idx] != 0)
                samples.push_back(gradient.data[idx].mag);
        }
    }

    const float hi =
        std::max(percentile(samples, 99.5f), 1e-12f);

    cv::Mat out(height, width, CV_8U, cv::Scalar(0));

    for (int y = 0; y < height; ++y)
    {
        uchar* dst = out.ptr<uchar>(y);

        for (int x = 0; x < width; ++x)
        {
            const size_t idx =
                static_cast<size_t>(z) * sliceSize +
                static_cast<size_t>(y) * width +
                static_cast<size_t>(x);

            dst[x] =
                cv::saturate_cast<uchar>(
                    255.0f * clamp01(gradient.data[idx].mag / hi)
                    );
        }
    }

    return out;
}

static cv::Mat floatMapTo8U(
    const cv::Mat& map,
    float highPercentile = 99.5f)
{
    CV_Assert(map.type() == CV_32F);

    std::vector<float> samples;
    samples.reserve(
        static_cast<size_t>((map.rows + 1) / 2) *
        static_cast<size_t>((map.cols + 1) / 2)
        );

    for (int y = 0; y < map.rows; y += 2)
    {
        const float* row = map.ptr<float>(y);
        for (int x = 0; x < map.cols; x += 2)
        {
            if (row[x] > 0.0f)
                samples.push_back(row[x]);
        }
    }

    const float hi =
        std::max(percentile(samples, highPercentile), 1e-8f);

    cv::Mat out(map.size(), CV_8U, cv::Scalar(0));

    for (int y = 0; y < map.rows; ++y)
    {
        const float* src = map.ptr<float>(y);
        uchar* dst = out.ptr<uchar>(y);

        for (int x = 0; x < map.cols; ++x)
        {
            dst[x] =
                cv::saturate_cast<uchar>(
                    255.0f * clamp01(src[x] / hi)
                    );
        }
    }

    return out;
}

static cv::Mat floatMapTo16U01(const cv::Mat& map)
{
    CV_Assert(map.type() == CV_32F);

    cv::Mat out(map.size(), CV_16U, cv::Scalar(0));

    for (int y = 0; y < map.rows; ++y)
    {
        const float* src = map.ptr<float>(y);
        uint16_t* dst = out.ptr<uint16_t>(y);

        for (int x = 0; x < map.cols; ++x)
        {
            dst[x] =
                static_cast<uint16_t>(
                    std::lround(65535.0f * clamp01(src[x]))
                    );
        }
    }

    return out;
}

static cv::Mat makeOverlay(
    const cv::Mat& display,
    const cv::Mat& mask,
    const cv::Vec3b& color)
{
    cv::Mat overlay;
    cv::cvtColor(display, overlay, cv::COLOR_GRAY2BGR);

    for (int y = 0; y < display.rows; ++y)
    {
        const uchar* m = mask.ptr<uchar>(y);
        cv::Vec3b* dst = overlay.ptr<cv::Vec3b>(y);

        for (int x = 0; x < display.cols; ++x)
        {
            if (m[x] != 0)
                dst[x] = color;
        }
    }

    return overlay;
}

static cv::Mat makeCombinedOverlay(
    const cv::Mat& display,
    const cv::Mat& standard,
    const cv::Mat& adaptive,
    const cv::Mat& phase)
{
    cv::Mat overlay;
    cv::cvtColor(display, overlay, cv::COLOR_GRAY2BGR);

    for (int y = 0; y < display.rows; ++y)
    {
        const uchar* s = standard.ptr<uchar>(y);
        const uchar* a = adaptive.ptr<uchar>(y);
        const uchar* p = phase.ptr<uchar>(y);
        cv::Vec3b* dst = overlay.ptr<cv::Vec3b>(y);

        for (int x = 0; x < display.cols; ++x)
        {
            const int code =
                (s[x] != 0 ? 1 : 0) |
                (a[x] != 0 ? 2 : 0) |
                (p[x] != 0 ? 4 : 0);

            switch (code)
            {
            case 1: // only ordinary Canny = red
                dst[x] = cv::Vec3b(0, 0, 255);
                break;
            case 2: // only adaptive Canny = green
                dst[x] = cv::Vec3b(0, 255, 0);
                break;
            case 3: // ordinary + adaptive = yellow
                dst[x] = cv::Vec3b(0, 255, 255);
                break;
            case 4: // only PC = blue
                dst[x] = cv::Vec3b(255, 0, 0);
                break;
            case 5: // ordinary + PC = magenta
                dst[x] = cv::Vec3b(255, 0, 255);
                break;
            case 6: // adaptive + PC = cyan
                dst[x] = cv::Vec3b(255, 255, 0);
                break;
            case 7: // all three = white
                dst[x] = cv::Vec3b(255, 255, 255);
                break;
            default:
                break;
            }
        }
    }

    return overlay;
}

static long long countMask(const cv::Mat& mask)
{
    return static_cast<long long>(cv::countNonZero(mask));
}

static double jaccard(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat inter;
    cv::Mat uni;
    cv::bitwise_and(a, b, inter);
    cv::bitwise_or(a, b, uni);

    const double i = static_cast<double>(cv::countNonZero(inter));
    const double u = static_cast<double>(cv::countNonZero(uni));

    return u > 0.0 ? i / u : 1.0;
}

static void createOutputFolders()
{
    fs::create_directories(OUTPUT_FOLDER);
    fs::create_directories(fs::path(OUTPUT_FOLDER) / "00_original");
    fs::create_directories(fs::path(OUTPUT_FOLDER) / "01_roi");
    fs::create_directories(fs::path(OUTPUT_FOLDER) / "02_gradient");
    fs::create_directories(fs::path(OUTPUT_FOLDER) / "03_canny_standard");
    fs::create_directories(fs::path(OUTPUT_FOLDER) / "04_canny_adaptive");
    fs::create_directories(fs::path(OUTPUT_FOLDER) / "05_phase_congruency_map");
    fs::create_directories(fs::path(OUTPUT_FOLDER) / "06_phase_congruency_nms");
    fs::create_directories(fs::path(OUTPUT_FOLDER) / "07_phase_congruency_edges");
    fs::create_directories(fs::path(OUTPUT_FOLDER) / "08_overlays");
}

// ============================================================
// MAIN
// ============================================================

int main()
{
    try
    {
        createOutputFolders();

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

        const size_t sliceSize =
            static_cast<size_t>(width) *
            static_cast<size_t>(height);

        const size_t total =
            sliceSize * static_cast<size_t>(depth);

        std::vector<uint8_t> seedROI3D;
        std::vector<uint8_t> broadROI3D;

        buildROIVolumes(
            volume,
            width,
            height,
            depth,
            seedROI3D,
            broadROI3D
            );

        // ----------------------------------------------------
        // DETECTORS 1 AND 2 SHARE THE EXACT SAME 3D GRADIENT + NMS
        // ----------------------------------------------------

        GradientVolume gradient =
            computeGradient3D(volume, width, height, depth);

        std::vector<uint8_t> nms = computeNMS3D(gradient);

        // Threshold statistics are taken from the tighter seed ROI so that
        // remote reconstruction texture cannot dominate the histogram.
        std::vector<float> globalNmsSamples =
            collectNmsGradientSamples(
                gradient,
                nms,
                seedROI3D,
                2
                );

        if (globalNmsSamples.empty())
        {
            std::cerr << "No 3D-NMS candidates inside the seed ROI."
                      << std::endl;
            return 1;
        }

        const float standardHigh =
            percentile(
                globalNmsSamples,
                STANDARD_HIGH_PERCENTILE
                );

        const float standardLow =
            standardHigh * STANDARD_LOW_HIGH_RATIO;

        float adaptiveGlobalHigh =
            otsuThresholdFloatSamples(
                globalNmsSamples,
                ADAPT_OTSU_BINS
                );

        if (!std::isfinite(adaptiveGlobalHigh) ||
            adaptiveGlobalHigh <= 1e-12f)
        {
            adaptiveGlobalHigh =
                percentile(globalNmsSamples, 80.0f);
        }

        std::cout
            << "\nOrdinary 3D Canny thresholds:\n"
            << "  high = " << standardHigh << '\n'
            << "  low  = " << standardLow << "\n\n"
            << "Adaptive-Canny global Otsu reference:\n"
            << "  high = " << adaptiveGlobalHigh << '\n'
            << "  low  = "
            << adaptiveGlobalHigh * ADAPT_LOW_HIGH_RATIO
            << std::endl;

        std::vector<uint8_t> cannyStandard =
            hysteresisGlobalGradient(
                gradient,
                nms,
                seedROI3D,
                broadROI3D,
                standardLow,
                standardHigh
                );

        AdaptiveThresholdGrid adaptiveGrid =
            buildAdaptiveThresholdGrid(
                gradient,
                nms,
                seedROI3D,
                adaptiveGlobalHigh
                );

        std::vector<uint8_t> cannyAdaptive =
            hysteresisAdaptiveGradient(
                gradient,
                nms,
                seedROI3D,
                broadROI3D,
                adaptiveGrid
                );

        // Save gradient previews now, before releasing the large gradient.
        if (SAVE_ALL_SLICE_DIAGNOSTICS)
        {
            for (int z = 0; z < depth; ++z)
            {
                const int globalZ = FIRST_SLICE + z;
                const std::string id = sliceName(globalZ);

                const cv::Mat gradientPreview =
                    gradientSlicePreview(gradient, broadROI3D, z);

                cv::imwrite(
                    (fs::path(OUTPUT_FOLDER) /
                     "02_gradient" /
                     ("gradient_" + id + ".png")).string(),
                    gradientPreview
                    );
            }
        }

        // The phase-congruency stage does not need the large 3D gradient.
        gradient.data.clear();
        gradient.data.shrink_to_fit();
        nms.clear();
        nms.shrink_to_fit();
        globalNmsSamples.clear();
        globalNmsSamples.shrink_to_fit();

        // ----------------------------------------------------
        // DETECTOR 3: KOVESI-STYLE NOISE-COMPENSATED PHASE CONGRUENCY
        // ----------------------------------------------------

        PhaseFilterBank phaseBank =
            buildPhaseFilterBank(height, width);

        std::vector<float> phaseNmsScore(total, 0.0f);
        std::vector<float> phaseScoreSamples;
        phaseScoreSamples.reserve(total / 80 + 1024);

        std::vector<float> phaseNoiseThresholds(
            static_cast<size_t>(depth),
            0.0f
            );

        for (int z = 0; z < depth; ++z)
        {
            const int globalZ = FIRST_SLICE + z;

            std::cout << "\rPhase congruency: slice "
                      << globalZ << " / " << LAST_SLICE
                      << std::flush;

            cv::Mat slice(
                height,
                width,
                CV_32F,
                volume.data() +
                    static_cast<size_t>(z) * sliceSize
                );

            const cv::Mat broadMask =
                maskSliceFromVolume(
                    broadROI3D,
                    width,
                    height,
                    z
                    );

            const cv::Mat seedMask =
                maskSliceFromVolume(
                    seedROI3D,
                    width,
                    height,
                    z
                    );

            PhaseCongruencyResult pc =
                computePhaseCongruency2D(
                    slice,
                    broadMask,
                    phaseBank
                    );

            phaseNoiseThresholds[static_cast<size_t>(z)] =
                pc.meanNoiseThreshold;

            float* dst =
                phaseNmsScore.data() +
                static_cast<size_t>(z) * sliceSize;

            for (int y = 0; y < height; ++y)
            {
                const float* src = pc.nmsScore.ptr<float>(y);
                const uchar* seedRow = seedMask.ptr<uchar>(y);

                for (int x = 0; x < width; ++x)
                {
                    const float v = src[x];
                    dst[static_cast<size_t>(y) * width + x] = v;

                    // Use only the tight object-neighbourhood distribution to
                    // select the final PC hysteresis threshold.
                    if (seedRow[x] != 0 && v > 0.0f)
                        phaseScoreSamples.push_back(v);
                }
            }

            if (SAVE_ALL_SLICE_DIAGNOSTICS)
            {
                const std::string id = sliceName(globalZ);

                const cv::Mat pcPreview =
                    floatMapTo8U(pc.pc, 99.5f);

                const cv::Mat stepnessPreview =
                    floatMapTo8U(pc.stepness, 99.5f);

                const cv::Mat edgeScorePreview =
                    floatMapTo8U(pc.edgeScore, 99.5f);

                const cv::Mat pcNmsPreview =
                    floatMapTo8U(pc.nmsScore, 99.5f);

                cv::imwrite(
                    (fs::path(OUTPUT_FOLDER) /
                     "05_phase_congruency_map" /
                     ("pc_" + id + ".png")).string(),
                    pcPreview
                    );

                cv::imwrite(
                    (fs::path(OUTPUT_FOLDER) /
                     "05_phase_congruency_map" /
                     ("stepness_" + id + ".png")).string(),
                    stepnessPreview
                    );

                cv::imwrite(
                    (fs::path(OUTPUT_FOLDER) /
                     "05_phase_congruency_map" /
                     ("pc_edge_score_" + id + ".png")).string(),
                    edgeScorePreview
                    );

                if (SAVE_PHASE_CONGRUENCY_16BIT)
                {
                    cv::imwrite(
                        (fs::path(OUTPUT_FOLDER) /
                         "05_phase_congruency_map" /
                         ("pc16_" + id + ".png")).string(),
                        floatMapTo16U01(pc.pc)
                        );

                    cv::imwrite(
                        (fs::path(OUTPUT_FOLDER) /
                         "05_phase_congruency_map" /
                         ("pc_edge_score16_" + id + ".png")).string(),
                        floatMapTo16U01(pc.edgeScore)
                        );
                }

                cv::imwrite(
                    (fs::path(OUTPUT_FOLDER) /
                     "06_phase_congruency_nms" /
                     ("pc_nms_" + id + ".png")).string(),
                    pcNmsPreview
                    );
            }
        }

        std::cout << std::endl;

        if (phaseScoreSamples.empty())
        {
            std::cerr << "Phase congruency produced no NMS candidates."
                      << std::endl;
            return 1;
        }

        const float pcHigh = std::max(
            PC_HIGH_FLOOR,
            percentile(phaseScoreSamples, PC_HIGH_PERCENTILE)
            );

        const float pcLow = std::max(
            PC_LOW_FLOOR,
            pcHigh * PC_LOW_HIGH_RATIO
            );

        std::cout
            << "\nPhase-congruency hysteresis thresholds:\n"
            << "  high = " << pcHigh << '\n'
            << "  low  = " << pcLow << std::endl;

        std::vector<uint8_t> phaseEdges =
            hysteresisFloatScore3D(
                phaseNmsScore,
                seedROI3D,
                broadROI3D,
                width,
                height,
                depth,
                pcLow,
                pcHigh
                );

        // ----------------------------------------------------
        // SAVE ALL DETECTOR MASKS + OVERLAYS + REPORT
        // ----------------------------------------------------

        std::ofstream report(
            fs::path(OUTPUT_FOLDER) / "report_three_detectors.txt"
            );

        report << std::fixed << std::setprecision(9);
        report
            << "THREE DETECTOR COMPARISON\n"
            << "=========================\n\n"
            << "Slice range\n"
            << "-----------\n"
            << "first  = " << FIRST_SLICE << '\n'
            << "center = " << CENTER_SLICE << '\n'
            << "last   = " << LAST_SLICE << '\n'
            << "depth  = " << depth << "\n\n"
            << "Volume\n"
            << "------\n"
            << "width  = " << width << '\n'
            << "height = " << height << '\n'
            << "spacing um = "
            << SPACING_X << ", "
            << SPACING_Y << ", "
            << SPACING_Z << "\n\n"
            << "Common Gaussian smoothing\n"
            << "-------------------------\n"
            << "sigma XY = " << SIGMA_XY << '\n'
            << "sigma Z  = " << SIGMA_Z << "\n\n"
            << "Detector 1: ordinary/global 3D Canny\n"
            << "------------------------------------\n"
            << "high percentile = "
            << STANDARD_HIGH_PERCENTILE << '\n'
            << "high threshold  = " << standardHigh << '\n'
            << "low/high ratio  = "
            << STANDARD_LOW_HIGH_RATIO << '\n'
            << "low threshold   = " << standardLow << "\n\n"
            << "Detector 2: adaptive Otsu 3D Canny\n"
            << "------------------------------------\n"
            << "tile size = " << ADAPT_TILE_SIZE << '\n'
            << "tile margin = " << ADAPT_TILE_MARGIN << '\n'
            << "Z radius = " << ADAPT_Z_RADIUS << '\n'
            << "global Otsu high = " << adaptiveGlobalHigh << '\n'
            << "low/high ratio = " << ADAPT_LOW_HIGH_RATIO << '\n'
            << "global floor ratio = " << ADAPT_GLOBAL_FLOOR_RATIO << '\n'
            << "global ceiling ratio = " << ADAPT_GLOBAL_CEIL_RATIO
            << "\n\n"
            << "Detector 3: Kovesi-style phase congruency\n"
            << "-----------------------------------------\n"
            << "scales = " << PC_NSCALE << '\n'
            << "orientations = " << PC_NORIENT << '\n'
            << "minimum wavelength = "
            << PC_MIN_WAVELENGTH << '\n'
            << "scale multiplier = " << PC_MULT << '\n'
            << "sigmaOnf = " << PC_SIGMA_ON_F << '\n'
            << "noise K = " << PC_NOISE_K << '\n'
            << "step weight floor = " << PC_STEP_WEIGHT_FLOOR << '\n'
            << "anisotropy weight floor = "
            << PC_ANISOTROPY_WEIGHT_FLOOR << '\n'
            << "PC high percentile = "
            << PC_HIGH_PERCENTILE << '\n'
            << "PC high floor = " << PC_HIGH_FLOOR << '\n'
            << "PC high threshold = " << pcHigh << '\n'
            << "PC low threshold  = " << pcLow << "\n\n"
            << "Per-slice comparison\n"
            << "--------------------\n"
            << "slice,standard_count,adaptive_count,pc_count,"
               "J_standard_adaptive,J_standard_pc,J_adaptive_pc,"
               "pc_mean_noise_threshold\n";

        for (int z = 0; z < depth; ++z)
        {
            const int globalZ = FIRST_SLICE + z;
            const std::string id = sliceName(globalZ);

            cv::Mat original;
            if (!loadFloatSlice(files[globalZ], original))
                return 1;

            const cv::Mat display = makeDisplayImage(original);

            const cv::Mat seedRoiMask =
                maskSliceFromVolume(
                    seedROI3D,
                    width,
                    height,
                    z
                    );

            const cv::Mat broadRoiMask =
                maskSliceFromVolume(
                    broadROI3D,
                    width,
                    height,
                    z
                    );

            const cv::Mat standardMask =
                maskSliceFromVolume(
                    cannyStandard,
                    width,
                    height,
                    z
                    );

            const cv::Mat adaptiveMask =
                maskSliceFromVolume(
                    cannyAdaptive,
                    width,
                    height,
                    z
                    );

            const cv::Mat phaseMask =
                maskSliceFromVolume(
                    phaseEdges,
                    width,
                    height,
                    z
                    );

            cv::imwrite(
                (fs::path(OUTPUT_FOLDER) /
                 "00_original" /
                 ("original_" + id + ".png")).string(),
                display
                );

            cv::imwrite(
                (fs::path(OUTPUT_FOLDER) /
                 "01_roi" /
                 ("seed_roi_" + id + ".png")).string(),
                seedRoiMask
                );

            cv::imwrite(
                (fs::path(OUTPUT_FOLDER) /
                 "01_roi" /
                 ("broad_roi_" + id + ".png")).string(),
                broadRoiMask
                );

            cv::imwrite(
                (fs::path(OUTPUT_FOLDER) /
                 "03_canny_standard" /
                 ("standard_" + id + ".png")).string(),
                standardMask
                );

            cv::imwrite(
                (fs::path(OUTPUT_FOLDER) /
                 "04_canny_adaptive" /
                 ("adaptive_" + id + ".png")).string(),
                adaptiveMask
                );

            cv::imwrite(
                (fs::path(OUTPUT_FOLDER) /
                 "07_phase_congruency_edges" /
                 ("pc_edges_" + id + ".png")).string(),
                phaseMask
                );

            const cv::Mat overlayStandard =
                makeOverlay(
                    display,
                    standardMask,
                    cv::Vec3b(0, 0, 255)
                    );

            const cv::Mat overlayAdaptive =
                makeOverlay(
                    display,
                    adaptiveMask,
                    cv::Vec3b(0, 255, 0)
                    );

            const cv::Mat overlayPC =
                makeOverlay(
                    display,
                    phaseMask,
                    cv::Vec3b(255, 0, 0)
                    );

            const cv::Mat overlayCombined =
                makeCombinedOverlay(
                    display,
                    standardMask,
                    adaptiveMask,
                    phaseMask
                    );

            cv::imwrite(
                (fs::path(OUTPUT_FOLDER) /
                 "08_overlays" /
                 ("standard_red_" + id + ".png")).string(),
                overlayStandard
                );

            cv::imwrite(
                (fs::path(OUTPUT_FOLDER) /
                 "08_overlays" /
                 ("adaptive_green_" + id + ".png")).string(),
                overlayAdaptive
                );

            cv::imwrite(
                (fs::path(OUTPUT_FOLDER) /
                 "08_overlays" /
                 ("pc_blue_" + id + ".png")).string(),
                overlayPC
                );

            cv::imwrite(
                (fs::path(OUTPUT_FOLDER) /
                 "08_overlays" /
                 ("combined_" + id + ".png")).string(),
                overlayCombined
                );

            report
                << globalZ << ','
                << countMask(standardMask) << ','
                << countMask(adaptiveMask) << ','
                << countMask(phaseMask) << ','
                << jaccard(standardMask, adaptiveMask) << ','
                << jaccard(standardMask, phaseMask) << ','
                << jaccard(adaptiveMask, phaseMask) << ','
                << phaseNoiseThresholds[static_cast<size_t>(z)]
                << '\n';
        }

        report.close();

        const long long totalStandard =
            static_cast<long long>(
                std::count(
                    cannyStandard.begin(),
                    cannyStandard.end(),
                    static_cast<uint8_t>(1)
                    )
                );

        const long long totalAdaptive =
            static_cast<long long>(
                std::count(
                    cannyAdaptive.begin(),
                    cannyAdaptive.end(),
                    static_cast<uint8_t>(1)
                    )
                );

        const long long totalPC =
            static_cast<long long>(
                std::count(
                    phaseEdges.begin(),
                    phaseEdges.end(),
                    static_cast<uint8_t>(1)
                    )
                );

        std::cout
            << "\nFinished.\n"
            << "Output folder:\n"
            << OUTPUT_FOLDER << "\n\n"
            << "Accepted 3D edge voxels:\n"
            << "  ordinary 3D Canny = " << totalStandard << '\n'
            << "  adaptive 3D Canny = " << totalAdaptive << '\n'
            << "  phase congruency   = " << totalPC << "\n\n"
            << "Combined overlay colours:\n"
            << "  red     = ordinary Canny only\n"
            << "  green   = adaptive Canny only\n"
            << "  blue    = phase congruency only\n"
            << "  yellow  = ordinary + adaptive\n"
            << "  magenta = ordinary + phase congruency\n"
            << "  cyan    = adaptive + phase congruency\n"
            << "  white   = all three detectors\n"
            << std::endl;

        return 0;
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "OpenCV exception:\n"
                  << e.what() << std::endl;
        return 2;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception:\n"
                  << e.what() << std::endl;
        return 3;
    }
}
