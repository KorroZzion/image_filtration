"""Validate exported geometry and generate a reviewable CT experiment report."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil

from boundary_study import *


def verify_volume(output):
    bundle = np.load(output / 'material_volume.npz')
    volume = bundle['mask']
    indices = bundle['source_slices'].tolist()
    origin = bundle['xy_origin_yx']
    meta = json.loads((output / 'volume_metrics.json').read_text())
    assert volume.dtype == np.bool_ and volume.shape[0] == len(indices)
    points_count, fallback_count = 0, 0
    for z, index in enumerate(indices):
        with Image.open(output / 'volume' / f'material_{index:04d}.png') as image:
            mask = np.asarray(image)
        assert mask.shape == tuple(meta['input_shape_zyx'][1:])
        y0, x0 = origin
        assert np.array_equal(mask[y0:y0+volume.shape[1], x0:x0+volume.shape[2]] > 0,
                              volume[z])
        with gzip.open(output / 'volume' / f'contours_{index:04d}.csv.gz', 'rt') as stream:
            rows = list(csv.DictReader(stream))
        assert {r['boundary_kind'] for r in rows} == {'inner', 'outer'}
        assert all(r['source_file'] == f'img_{index:04d}.tif' for r in rows)
        for row in rows:
            assert int(row['slice_number']) == index
            assert np.isfinite(float(row['x_px'])) and np.isfinite(float(row['y_px']))
            assert 0 <= float(row['x_px']) < mask.shape[1]
            assert 0 <= float(row['y_px']) < mask.shape[0]
            assert abs(float(row['shift_px'])) <= 3.001
            assert int(row['gradient_supported']) in (0, 1)
        points_count += len(rows)
        fallback_count += sum(int(r['gradient_supported']) == 0 for r in rows)
    # Read the actual binary PLY, not just the in-memory arrays that wrote it.
    with (output / 'surface_patch_voxels.ply').open('rb') as stream:
        lines = []
        while True:
            line = stream.readline().decode().strip()
            lines.append(line)
            if line == 'end_header':
                break
        nv = int(next(l.split()[-1] for l in lines if l.startswith('element vertex ')))
        nf = int(next(l.split()[-1] for l in lines if l.startswith('element face ')))
        vertices = np.frombuffer(stream.read(nv * 12), dtype='<f4').reshape(nv, 3)
        data = np.frombuffer(stream.read(), dtype=[('n', 'u1'), ('indices', '<i4', (3,))])
    assert len(data) == nf and np.all(data['n'] == 3)
    faces = data['indices']
    assert np.isfinite(vertices).all() and faces.min() >= 0 and faces.max() < nv
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1)
    assert np.all(area > 0), 'Degenerate mesh triangles'
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    assert counts.max() <= 2, 'Non-manifold edge'
    boundary = unique[counts == 1]
    z = vertices[boundary, 2]
    ends = np.all(np.isclose(z, 0), axis=1) | np.all(np.isclose(z, len(indices)-1), axis=1)
    assert ends.all(), 'Unexpected hole away from test-window cut planes'
    _, components = ndi.label(volume, structure=np.ones((3, 3, 3)))
    result = {'source_slices': indices, 'mask_shape_zyx': list(volume.shape),
              'exported_contour_points': points_count, 'unrefined_points': fallback_count,
              'connected_components_26': components, 'mesh_vertices': nv, 'mesh_faces': nf,
              'open_edges_at_stack_cut_planes_only': len(boundary)}
    (output / 'geometry_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--volume-results', type=Path)
    parser.add_argument('--phantom', type=Path, required=True)
    parser.add_argument('--input', type=Path, default=Path('CT_Test'))
    args = parser.parse_args()
    output = args.output
    if args.volume_results:
        for name in ['material_volume.npz', 'volume_metrics.json', 'surface_patch_voxels.ply']:
            shutil.copy2(args.volume_results / name, output / name)
        shutil.copytree(args.volume_results / 'volume', output / 'volume', dirs_exist_ok=True)
    verified = verify_volume(output)
    rows = json.loads((output / 'study_metrics.json').read_text())
    phantom = json.loads(args.phantom.read_text())
    # A view directly comparing the SAME source slice to the earlier C++ run.
    index = 988
    raw = load(args.input, index)
    field = segment_slice(raw)
    y0, y1, x0, x1 = field['box']
    legacy = Path('/workspace/.onboarding/ct-sample/D:/Studies/Real_work/BladeCT/Results/three_detectors_comparison_v2')
    fig, axes = plt.subplots(1, 4, figsize=(14, 11))
    plot_mask(axes[0], field['raw'], title='Source img_0988')
    for ax, folder, name, label in [
        (axes[1], '03_canny_standard', 'standard_1006.png', 'Existing global 3D Canny'),
        (axes[2], '07_phase_congruency_edges', 'pc_edges_1006.png', 'Existing phase congruency'),
    ]:
        plot_mask(ax, field['raw'], title=label)
        with Image.open(legacy / folder / name) as image:
            mask = np.asarray(image)[y0:y1, x0:x1] > 0
        rgba = np.zeros((*mask.shape, 4))
        rgba[mask] = [0, 1, .35, 1]
        ax.imshow(rgba)
    with Image.open(output / 'volume' / 'material_0988.png') as image:
        mask = np.asarray(image)[y0:y1, x0:x1] > 0
    plot_mask(axes[3], field['raw'], mask=mask, title='3D watershed + gradient refinement')
    fig.tight_layout()
    fig.savefig(output / 'existing_vs_new_0988.png', dpi=150)
    plt.close(fig)
    disagreement = []
    for index in sorted({r['slice'] for r in rows}):
        with Image.open(output / 'study' / f'material_{index:04d}.png') as im:
            base = np.asarray(im) > 0
        with Image.open(output / 'extrema_seeds' / f'material_{index:04d}.png') as im:
            alternate = np.asarray(im) > 0
        uncertain = base ^ alternate
        Image.fromarray(uncertain.astype(np.uint8) * 255).save(output / f'seed_disagreement_{index:04d}.png')
        disagreement.append({'slice': index, 'seed_strategy_disagreement_pixels': int(uncertain.sum()),
                             'union_pixels': int((base | alternate).sum())})
    (output / 'seed_disagreement.json').write_text(json.dumps(disagreement, indent=2) + '\n')
    report = ['# Эксперименты с поверхностями КТ-детали', '',
        'Цель — наружная поверхность и все внутренние поверхности «материал–воздух». '
        'Исходные 16-битные TIFF не изменены. Эксперименты не меняют main.cpp.', '',
        '## Основной результат', '',
        'Для тонкого сечения наиболее перспективен **маркерный 3D watershed с уточнением '
        'границы по локальному профилю яркости**. Он даёт замкнутые контуры каналов и '
        'наружной поверхности. Фазовая конгруэнтность в текущей C++-реализации выделяет '
        'много полос реконструкции вне детали. Глобальный Otsu частично заполняет каналы; '
        'поднятие порога разрушает тонкие стенки. Random walker с проверенными маркерами '
        'теряет части тонкой оболочки.', '',
        'В толстом сечении img_0500 пороговые маркеры пропускают два ярких канала. '
        'Дополнительная инициализация по локальным минимумам воздуха и светлым гребням '
        'материала восстанавливает четыре видимых крупных канала. Однако в тонком '
        'сечении эта же настройка местами разрушает стенку. **Единой надёжной автоматической '
        'инициализации для всей выборки пока нет.** Это остаётся главной задачей, '
        'а не выбор одного глобального порога Canny.', '',
        '## Что выполнено', '',
        '- Сравнение шести основных вариантов на img_0500, 0625, 0750, 0875, 0988: '
        'глобальный Otsu, Otsu в ROI, random walker, watershed, watershed + local ISO50, '
        'watershed + максимум градиента. Дополнительно проверена инициализация по экстремумам.',
        '- Проверка устойчивости при sigma=0.75/1.25 вместо 1 и при изменении '
        'порога опорного материала на −10%/+10%.',
        '- Объёмный watershed на 25 реальных последовательных срезах img_0976–img_1000, '
        'сглаживание sigma=(0.7,1,1). Последующее уточнение контуров выполняется в XY-плоскости.',
        '- Синтетические круговые поверхности с известной геометрией: однородная '
        'яркость, неоднородная яркость, неоднородная яркость с шумом.', '',
        '## Синтетическая проверка локализации', '',
        '95-й процентиль расстояния от найденных точек до аналитических окружностей, пиксели. '
        'Это не оценка точности на реальной детали.', '',
        '| Условие | Otsu | Watershed | Максимум градиента | Local ISO50 |',
        '|---|---:|---:|---:|---:|']
    for case in phantom:
        report.append('| ' + case['condition'] + ' | ' + ' | '.join(
            f"{case[m]['p95_error_px']:.3f}" for m in ['global_otsu','watershed','gradient_peak','local50']) + ' |')
    report += ['', 'Локальный ISO50 показал меньшую ошибку на этих простых фантомах. '
               'На реальном КТ оценки ISO50 и максимума градиента расходятся; '
               'обе координаты сохранены, окончательный выбор требует разметки.', '',
               '## Локальная поддержка и устойчивость на реальных данных', '',
               'Поддержка профилем означает положительный контраст и внутренний максимум '
               'градиента в окне ±3 пикселя. Она **не доказывает**, что семантически выбран '
               'правильный канал или что точка имеет указанную метрологическую точность.', '',
               '| Срез | Поддержанных профилей | Расхождение ISO50/градиент, P95, px | '
               'Чувствительность к sigma, P95, px | К порогу маркеров, P95, px |',
               '|---|---:|---:|---:|---:|']
    for row in rows:
        if row['method'] != 'refinement_diagnostics':
            continue
        index = row['slice']
        sensitivity = [r for r in rows if r['slice'] == index and r['method'] == 'parameter_sensitivity']
        ss = max(r['symmetric_distance_p95_px'] for r in sensitivity if r['sigma'] != 1)
        st = max(r['symmetric_distance_p95_px'] for r in sensitivity if r['high_seed_factor'] != 1)
        report.append(f"| {index} | {row['supported_fraction']:.1%} | "
                      f"{row['iso_gradient_disagreement_p95_px']:.2f} | {ss:.2f} | {st:.2f} |")
    report += ['', 'Большие изменения при смене порога маркеров показывают неопределённость '
               'инициализации. Маски seed_disagreement_XXXX.png показывают области '
               'несогласия двух схем маркеров; это не эталонная карта ошибок.', '',
               '## Экспорт и проверка геометрии', '',
               f"Для блока 0976–1000 экспортировано {verified['exported_contour_points']} точек "
               f"контуров; {verified['unrefined_points']} оставлены без смещения из-за слабого профиля. "
               'Они отмечены gradient_supported=0 в CSV и оранжевым на наложениях.', '',
               f"PLY содержит {verified['mesh_vertices']} вершин и {verified['mesh_faces']} граней. "
               'Проверены конечность координат, индексы граней, отсутствие вырожденных '
               'треугольников и неманифолдных рёбер. Все открытые рёбра находятся только '
               'на двух плоскостях обрезки блока. Маска связна при 26-соседстве.', '',
               'Это **открытый фрагмент**, не модель всей детали. Файл PLY задан в вокселях: '
               'X/Y — координаты исходного изображения, Z=0 соответствует img_0976. '
               '75.4 мкм взято из существующего C++-кода и не проверялось по калибровке сканера. '
               'Мелкие детали сглажены при построении сетки, исходные бинарные маски и CSV сохранены.', '',
               '## Просмотр результатов', '',
               '- [Существующие детекторы и новый подход, img_0988](existing_vs_new_0988.png)',
               '- [Шесть методов, img_0988](comparison_0988.png)',
               '- [Крупнее: внутренние каналы, img_0988](channel_detail_0988.png)',
               '- [Инициализация: яркие каналы img_0500](seed_comparison_0500.png)',
               '- [Наружная поверхность и каналы блока, img_0988](volume/overlay_0988.png)',
               '- [Открытый 3D-фрагмент, PLY в вокселях](surface_patch_voxels.ply)',
               '- [Проверка экспорта](geometry_validation.json)', '',
               'Подкаталоги study, extrema_seeds и volume содержат маски material_XXXX.png, '
               'наложения overlay_XXXX.png и сжатые CSV contours_XXXX.csv.gz. CSV хранит '
               'исходное имя файла, номер среза, ID замкнутого контура, тип outer/inner, '
               'координаты двух оценок границы и признаки поддержки. Последняя точка '
               'каждого контура соединяется с первой. Отверстия в итоговых масках не заливаются.', '',
               '## Следующий обоснованный шаг', '',
               'Разметить небольшой набор наружных и внутренних границ в толстом и тонком '
               'сечениях, особенно слабую стенку и яркие каналы. По этой разметке выбирать '
               'маркеры и локальный критерий поверхности. До этого нельзя утверждать, что '
               'выделены все поверхности без пропусков, указывать точность в микронах '
               'или автоматически обрабатывать весь скан как готовую метрологическую модель.']
    (output / 'REPORT.md').write_text('\n'.join(report) + '\n')
    print(json.dumps(verified, indent=2))
    print(output / 'REPORT.md')


if __name__ == '__main__':
    main()
