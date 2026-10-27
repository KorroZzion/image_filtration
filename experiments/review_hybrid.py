"""Verify the streamed sample masks and document the hybrid's actual outcome."""
import argparse
import json
from pathlib import Path

from boundary_study import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('CT_Test'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--study-results', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.output / 'hybrid_metrics.json').read_text())
    indices = [r['slice'] for r in rows]
    assert indices == list(range(500,1001)), 'Incomplete selected sample'
    files = sorted((args.output/'masks').glob('material_*.png'))
    assert [p.name for p in files] == [f'material_{i:04d}.png' for i in indices]
    for path, row in zip(files,rows):
        with Image.open(path) as image:
            mask = np.asarray(image)
        assert mask.shape == (1804,910) and mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0,255})
        assert int(np.count_nonzero(mask)) == row['material_pixels'] > 0
    figures = []
    for index in [500,625,750,875,988]:
        raw = load(args.input,index)
        field = segment_slice(raw,run_watershed=False)
        y0,y1,x0,x1 = field['box']
        fig,axes=plt.subplots(1,4,figsize=(14,11))
        plot_mask(axes[0],field['raw'],title=f'Source img_{index:04d}')
        for ax,path,title in [
            (axes[1],args.study_results/'study'/f'material_{index:04d}.png','Independent threshold seeds'),
            (axes[2],args.study_results/'extrema_seeds'/f'material_{index:04d}.png','Independent extrema seeds'),
            (axes[3],args.output/'refined_examples'/f'material_{index:04d}.png','Hybrid: exterior reset + channel seeds')]:
            with Image.open(path) as image:
                mask=np.asarray(image)[y0:y1,x0:x1]>0
            plot_mask(ax,field['raw'],mask=mask,title=title)
        fig.tight_layout()
        name=f'hybrid_comparison_{index:04d}.png'
        fig.savefig(args.output/name,dpi=145);plt.close(fig)
        figures.append(name)
    summary = {'completed_real_slices':len(indices),'first_source':'img_0500.tif',
               'last_source':'img_1000.tif','raw_mask_dimensions_xy':[910,1804],
               'seed_source':'img_0988.tif','images_verified':len(files),
               'minimum_material_pixels':min(r['material_pixels'] for r in rows),
               'maximum_material_pixels':max(r['material_pixels'] for r in rows),
               'sample_rows':[r for r in rows if r['slice'] in [500,625,750,875,988,1000]]}
    (args.output/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    text = ['# Гибридное прослеживание внутренних каналов: исследовательский опыт','',
            '**Вариант не принят для автоматического поиска всех поверхностей.** '
            'При просмотре img_0875 обнаружены пропуски видимых каналов: перенос '
            'с проверкой контраста преждевременно удаляет часть опорных областей, '
            'а новая полость без маркера не появляется. Маски ниже — результаты '
            'опыта для диагностики, не готовая сегментация всей детали.','',
            'Наружная опорная оболочка определяется заново по каждому срезу; '
            'из предыдущего среза передаются маркеры внутренних каналов и материала. '
            'Маркеры канала проверяются по текущему контрасту с окружающей стенкой, '
            'чтобы не удерживать полость после её закрытия.','',
            '**Обработаны все 501 доступный исходный TIFF, img_0500–img_1000.** '
            'Все 501 бинарная маска прочитаны обратно и проверены: исходные размеры, '
            'значения 0/255, ненулевой материал и совпадение числа пикселей с метриками. '
            'Это проверка результата выполнения, не эталонная оценка точности.','',
            'Начальная маска взята из 3D watershed на img_0988. Ни исходные TIFF, '
            'ни main.cpp не изменялись. Итоговые маски всех срезов — в masks; '
            'уточнённые маски, наложения и CSV для шести примеров — в refined_examples.','',
            '| Срез | Пикселей материала | Замкнутых полостей ≥25 px² |','|---|---:|---:|']
    for row in summary['sample_rows']:
        text.append(f"| {row['slice']} | {row['material_pixels']} | {row['holes_ge_25px']} |")
    text += ['','Число замкнутых полостей зависит от сечения и от открытых каналов, '
             'соединённых с внешним воздухом. Оно не является числом физических каналов '
             'во всей детали и не доказывает полноту сегментации.','',
             '## Просмотр','']
    for index,name in zip([500,625,750,875,988],figures):
        text.append(f'- [Независимые методы и гибрид, img_{index:04d}]({name})')
    text += ['','## Ограничения','',
             'Перенос только внутренних маркеров убирает зависимость наружной '
             'опорной оболочки от предыдущих срезов, но не устраняет все ошибки. '
             'Новая изолированная полость без исходного маркера может быть пропущена; '
             'сложные изменения связности требуют повторной инициализации. '
             'Тонкие стенки, яркие каналы и слабые переходы нужно проверить по наложениям. '
             'Для оценки точности и полноты нужна разметка реального скана.','',
             'Вариант переноса всей маски из track_boundaries.py был отклонён после '
             '488 переходов: на img_0500 накопился сильный дрейф наружной границы. '
             'Его прямой цикл остановлен и не считается успешно проверенным.']
    (args.output/'REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
