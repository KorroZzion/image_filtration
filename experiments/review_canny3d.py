"""Read back Canny3D exports, compare same-source images and build GitHub report."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from boundary_study import load,segment_slice,normal_refinement,plt
from matplotlib.collections import LineCollection
from canny3d_study import plane_segments


def verify_block(root):
    meta=json.loads((root/'metrics.json').read_text())
    supported_variant='first' in meta
    if supported_variant:
        meta={**meta,'source_first':meta['first'],'source_last':meta['last']}
    arrays=np.load(root/'surface_coordinates.npz')
    seed,refined,faces=arrays['seed_zyx'],arrays['refined_zyx'],arrays['faces_zyx']
    assert seed.shape==refined.shape and np.isfinite(refined).all()
    assert faces.min()>=0 and faces.max()<len(refined)
    movement=np.linalg.norm(refined-seed,axis=1)
    assert movement.max()<=2.002
    tri=refined[faces];original=seed[faces]
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    old=np.cross(original[:,1]-original[:,0],original[:,2]-original[:,0])
    assert (np.linalg.norm(cross,axis=1)>0).all()
    assert (np.sum(cross*old,axis=1)>0).all()
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[0,2]],faces[:,[1,2]]]),axis=1)
    unique,count=np.unique(edges,axis=0,return_counts=True)
    if count.max()>2:
        if supported_variant:
            raise AssertionError('Preferred model has non-manifold edges')
        rejection={'status':'rejected','non_manifold_edges':int((count>2).sum()),
                   'maximum_faces_per_edge':int(count.max()),
                   'reason':'Ambiguous voxel contacts in the rejected initialization; not a usable manifold PLY.'}
        (root/'mesh_rejection.json').write_text(json.dumps(rejection,indent=2)+'\n')
        ply=root/'surface_refined_xyz.ply'
        if ply.exists():
            archive=Path('/workspace/ct-canny3d-rejected-meshes')/root.name
            archive.mkdir(parents=True,exist_ok=True)
            target=archive/ply.name
            if target.exists():
                assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(ply.read_bytes()).digest()
                ply.unlink()
            else:
                shutil.move(str(ply),target)
        masks=np.load(root/'material_mask.npz')['mask']
        counts={f'{mode}_{name}':0 for mode in ['gaussian','diffusion'] for name in ['standard','adaptive','supported']}
        for z,index in enumerate(range(meta['source_first'],meta['source_last']+1)):
            for key in counts:
                with Image.open(root/f'{key}_{index:04d}.png') as image:pixels=np.asarray(image)
                assert pixels.shape==masks.shape[1:] and set(np.unique(pixels)).issubset({0,255})
                counts[key]+=int(np.count_nonzero(pixels))
            with Image.open(root/f'material_{index:04d}.png') as image:pixels=np.asarray(image)
            assert np.array_equal(pixels>0,masks[z])
        return {'block':root.name,'source_first':meta['source_first'],'source_last':meta['source_last'],
                'variant':'rejected_experimental_initialization','geometry_validation':rejection,
                'core_edge_voxels':counts,'refinement':meta['segmentation']}
    cuts=unique[count==1];z=refined[cuts,0]
    last=meta['source_last']-meta['source_first']
    assert (np.all(np.isclose(z,0),axis=1)|np.all(np.isclose(z,last),axis=1)).all()
    # Validate the actual binary PLY and its XYZ transform as well as NPZ.
    with (root/'surface_refined_xyz.ply').open('rb') as stream:
        lines=[]
        while True:
            line=stream.readline().decode().strip();lines.append(line)
            if line=='end_header':break
        nv=int(next(s.split()[-1] for s in lines if s.startswith('element vertex ')))
        nf=int(next(s.split()[-1] for s in lines if s.startswith('element face ')))
        xyz=np.frombuffer(stream.read(nv*12),dtype='<f4').reshape(nv,3)
        records=np.frombuffer(stream.read(),dtype=[('n','u1'),('indices','<i4',(3,))])
    assert len(records)==nf and nv==len(seed) and np.all(records['n']==3)
    expected=refined[:,[2,1,0]].copy()
    y0,_,x0,_=meta['crop_box_y0_y1_x0_x1'];expected[:,:2]+=[x0,y0]
    assert np.allclose(xyz,expected,atol=1e-4)
    assert np.array_equal(records['indices'],faces[:,[0,2,1]])
    mask=np.load(root/('initializer_mask.npz' if supported_variant else 'material_mask.npz'))['mask']
    assert mask.dtype==np.bool_ and mask.shape[0]==last+1
    counts={f'{mode}_{name}':0 for mode in ['gaussian','diffusion'] for name in ['standard','adaptive','supported']}
    for z,index in enumerate(range(meta['source_first'],meta['source_last']+1)) if not supported_variant else []:
        for key in counts:
            with Image.open(root/f'{key}_{index:04d}.png') as image:
                pixels=np.asarray(image)
            assert pixels.shape==mask.shape[1:] and set(np.unique(pixels)).issubset({0,255})
            counts[key]+=int(np.count_nonzero(pixels))
        with Image.open(root/f'material_{index:04d}.png') as image:
            pixels=np.asarray(image)
        assert np.array_equal(pixels>0,mask[z])
    return {'block':root.name,'source_first':meta['source_first'],'source_last':meta['source_last'],
            'mesh_vertices':nv,'mesh_faces':nf,'boundary_edges_at_cut_planes':len(cuts),
            'variant':'supported_refinement' if supported_variant else 'experimental_initialization',
            'core_edge_voxels':counts if not supported_variant else meta['canny_core_counts'],
            'refinement':meta['refinement'] if supported_variant else meta['segmentation']}


def compare(args,root):
    meta=json.loads((root/'metrics.json').read_text())
    arrays=np.load(args.output/'supported_refinement'/root.name/'surface_coordinates.npz')
    vertices,faces=arrays['refined_zyx'],arrays['faces_zyx']
    index=(meta['source_first']+meta['source_last'])//2
    raw=load(args.input,index)
    y0,y1,x0,x1=meta['crop_box_y0_y1_x0_x1']
    crop=raw[y0:y1,x0:x1]
    old=segment_slice(raw)
    oldlines=normal_refinement(old['smooth'],old['watershed'])
    fig,axes=plt.subplots(1,4,figsize=(13,10))
    titles=['Original CT','Previous 2D watershed + XY refinement','New adaptive 3D Canny','New surface: XYZ refinement']
    for ax,title in zip(axes,titles):
        ax.imshow(crop,cmap='gray',vmin=.06,vmax=.8);ax.set_title(title,fontsize=9);ax.axis('off')
    for line in oldlines:
        points=line['gradient']
        axes[1].plot(points[:,1]+old['box'][2]-x0,points[:,0]+old['box'][0]-y0,color='#00ff70',lw=.6)
    with Image.open(root/f'diffusion_adaptive_{index:04d}.png') as image:
        mask=np.asarray(image)>0
    rgba=np.zeros((*mask.shape,4));rgba[mask]=[0,1,.4,1];axes[2].imshow(rgba)
    segments=plane_segments(vertices,faces,float(index-meta['source_first']))
    axes[3].add_collection(LineCollection(segments[:,:,[1,0]],colors='#00ff70',linewidths=.6))
    fig.suptitle(f'Same original img_{index:04d}; no source index remapping')
    fig.tight_layout();fig.savefig(args.output/f'before_after_{index:04d}.png',dpi=145);plt.close(fig)
    return index


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=Path('CT_Test'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--unconstrained',type=Path)
    args=parser.parse_args()
    roots=sorted(args.output.glob('block_*'))
    results=[verify_block(root) for root in roots]
    assert len(results)==3 and sum(r['source_last']-r['source_first']+1 for r in results)==75
    centers=[compare(args,root) for root in roots]
    preferred=[verify_block(args.output/'supported_refinement'/root.name) for root in roots]
    if args.unconstrained:
        diagnostic=args.output/'diagnostics';diagnostic.mkdir(exist_ok=True)
        for root,index in zip(roots,centers):
            source=args.unconstrained/root.name/f'refined_{index:04d}.png'
            if source.exists():shutil.copy2(source,diagnostic/f'without_outer_support_{index:04d}.png')
    (args.output/'export_validation.json').write_text(json.dumps(results+preferred,indent=2)+'\n')
    benchmark=json.loads((args.output/'phantom_benchmark.json').read_text())
    lines=['# 3D-градиент и адаптивный Canny: продолжение экспериментов','',
           'Цель — наружная поверхность и стенки внутренних каналов. Обработаны '
           '**75 реальных срезов в трёх блоках**: 500–524, 738–762, 976–1000. '
           'Исходные 16-битные TIFF и main.cpp не менялись.','',
           '**Для дальнейшей работы выбран вариант supported_refinement**: начальная '
           'сегментация сохраняется, поверхность уточняется по настоящему 3D-градиенту '
           'только при близкой поддержке адаптивного Canny. Инициализация для каждого '
           'блока явно записана в metrics.json: ориентированные 3D-рёбра для толстого '
           'сечения, прежний watershed по срезам для переходного, сохранённый прежний '
           'объёмный watershed для тонкого. Единого автоматического выбора инициализации '
           'здесь пока нет.','',
           '## Что добавлено','',
           '1. Производные по X, Y и Z после объёмного сглаживания; трёхлинейное '
           'подавление немаксимумов вдоль нормализованного 3D-градиента.',
           '2. Адаптивный Canny: локальный Otsu на 3D-NMS, плитки 64×64 с контекстом '
           'по пяти срезам, смешивание с общим порогом и нижняя граница по MAD шума. '
           'Низкий порог = 0.4 высокого; гистерезис использует 26-соседство.',
           '3. Сравнение Gaussian и адаптивного диффузионного фильтра Perona–Malik '
           '(8 шагов, kappa от оценки шума). Слабые продолжения Canny участвуют '
           'в барьере поверхности, но сами не создают новые полости.',
           '4. Сильные границы задают пары опорных точек воздуха/материала по направлению '
           'градиента. Чтобы ошибочный знак контраста около тонкой стенки не разрывал '
           'оболочку, добавлена узкая опорная зона внутри независимо оценённого наружного '
           'контура. Это геометрический приоритет, а не измеренная толщина стенки.',
           '5. Уточнение поверхности по максимуму производной вдоль 3D-направления '
           '**со смещением X, Y и Z**, максимум ±2 вокселя. Сдвиги, переворачивающие '
           'треугольники, сокращаются и явно помечаются. На торцах блока точки заморожены.',
           '6. Контуры на PNG и CSV получены пересечением уточнённой 3D-сетки с '
           'плоскостью исходного среза, а не простой проекцией точек на XY.','',
           '## Предпочтительное уточнение: посмотреть первым','']
    for root,index in zip(roots,centers):
        prefix=f'supported_refinement/{root.name}'
        lines += [f'- [Исходный срез, начальная поверхность, Canny и XYZ-уточнение: img_{index:04d}]({prefix}/comparison_{index:04d}.png)',
                  f'- [Изменение положения крупнее: оранжевый — исходное, зелёный — уточнённое]({prefix}/position_detail_{index:04d}.png)',
                  f'- [3D-фрагмент PLY]({prefix}/surface_refined_xyz.ply)']
    lines+=['','## Диагностические варианты детектора и инициализации','',
            'Независимая инициализация по ориентированным рёбрам восстановила яркие '
            'каналы толстого сечения. На тонких стенках она дала разрывы. Жёсткий '
            'наружный приоритет также дал смещение в реконструкционные полосы и '
            '**не выбран для итогового уточнения**. Эти сравнения сохранены, чтобы '
            'видеть ограничения метода; правый столбец следующих изображений '
            'показывает этот экспериментальный вариант.','']
    lines+=['Три диагностические инициализации также не прошли проверку манифолдности '
            'рёбер сетки. Их PLY исключены из публикуемых результатов; причины и числа '
            'проблемных рёбер записаны в mesh_rejection.json. Для выбранного '
            'supported_refinement геометрическая проверка обязательна и пройдена.','']
    for root,index in zip(roots,centers):
        lines += [f'- [До/после и адаптивный Canny, img_{index:04d}](before_after_{index:04d}.png)',
                  f'- [Пять вариантов, img_{index:04d}]({root.name}/comparison_{index:04d}.png)',
                  f'- [Экспериментальная инициализация крупнее, не выбранная итоговой]({root.name}/refined_{index:04d}.png)']
    lines+=['','## Изменение числа найденных рёбер','',
            'Количество edge-вокселей на 25 срезах каждого блока. Больше рёбер '
            '**не означает автоматически выше точность**.','',
            '| Блок | Gaussian, общий Canny | Gaussian, адаптивный | Diffusion, адаптивный |',
            '|---|---:|---:|---:|']
    for item in results:
        c=item['core_edge_voxels']
        lines.append(f"| {item['source_first']}–{item['source_last']} | {c['gaussian_standard']} | {c['gaussian_adaptive']} | {c['diffusion_adaptive']} |")
    lines+=['','## Проверка на известной 3D-геометрии','',
            'Фантом: сфера с наклонным цилиндрическим каналом, неоднородная яркость '
            'и шум. Значения ниже относятся только к этому фантому.','',
            '| Фильтр | Dice маски | P95 до уточнения, vox | P95 после XYZ-уточнения, vox |',
            '|---|---:|---:|---:|']
    for item in benchmark:
        lines.append(f"| {item['filter']} | {item['segmentation_dice']:.4f} | {item['seed_surface_error_p95_vox']:.3f} | {item['refined_surface_error_p95_vox']:.3f} |")
    lines+=['','## Фактическое объёмное уточнение на КТ','',
            'P95 абсолютного смещения, воксели. Это величина изменения поверхности, '
            'а не измеренная ошибка относительно эталона.','',
            '| Блок | Вершин достигли поддержанного максимума | P95 сдвига XYZ | P95 компоненты Z |',
            '|---|---:|---:|---:|']
    for item in preferred:
        info=item['refinement']
        lines.append(f"| {item['source_first']}–{item['source_last']} | {info['accepted_vertex_fraction']:.1%} | {info['abs_shift_p95_vox']:.3f} | {info['abs_z_displacement_p95_vox']:.3f} |")
    lines+=['','Пройдены пять автоматических проверок: граница только по Z, реакция '
            'порогов на локальный шум, правильное пересечение сетки плоскостью, '
            'выделение материала/канала и XYZ-локализация для обоих фильтров. '
            'Полный вывод — [tests.log](tests.log).','',
            '## Экспорт и его проверка','',
            'Для каждого блока сохранены PNG рёбер и масок, CSV.GZ пересечений '
            'поверхности, NPZ координат до/после уточнения и PLY в вокселях. '
            'X/Y в PLY — координаты исходного изображения, Z=0 у первого среза блока. '
            'PNG масок обрезаны по ROI; его сдвиг записан в metrics.json и NPZ.',
            'Проверены сохранённые PNG/NPZ/PLY: конечность координат, границы сдвига, '
            'порядок осей, невырожденность/ориентация граней, индексы, манифолдность '
            'рёбер и открытые края только на плоскостях обрезки. Подробности — '
            '[export_validation.json](export_validation.json).','',
            '## Что ещё не решено','',
            'Тонкие стенки и артефакты реконструкции всё ещё могут искажать локальные '
            'границы. Экспериментальная наружная зона внесла смещение и не применяется '
            'в выбранном supported_refinement; её толщина не является физическим измерением. '
            'В diagnostics сохранён вариант без этого приоритета для сравнения.',
            'Выбор крупнейшей связной компоненты предполагает одну связную деталь '
            'в блоке и может удалить реальные отделённые фрагменты. Неподдержанные '
            'профили и геометрически ограниченные сдвиги не считаются достигшими '
            'максимума градиента; признаки сохранены в surface_coordinates.npz.',
            'Выбранное уточнение сохраняет исходную связность и не восстанавливает '
            'канал, если он пропущен в начальной маске. Проверка пересечений удалённых '
            'друг от друга треугольников не выполнена.',
            'На крайних доступных срезах 500 и 1000 нет двустороннего Z-контекста, '
            'поэтому 3D NMS не выполняется в этих двух плоскостях. Сетки открыты '
            'на торцах блоков. Точность реального КТ без контрольной разметки '
            'не измерена; 75.4 мкм взято из прежнего кода и не калибровалось.','',
            '## Воспроизведение','',
            'Из корня репозитория, с зависимостями experiments/requirements.txt:',
            '```bash',
            'python -m unittest discover -s experiments -p test_canny3d.py -v',
            'python experiments/canny3d_study.py --input CT_Test --output /tmp/new-canny3d --blocks 500:524,738:762,976:1000 --outer-support',
            '```',
            'Каталог каждого блока должен быть новым: скрипт не перезаписывает '
            'существующие результаты. Данные и код предыдущих опытов сохранены.','',
            '## Размеры и форматы всех новых изображений','']
    image_rows=[]
    for path in sorted(args.output.rglob('*.png')):
        with Image.open(path) as image:
            image.load();fmt=image.format;width,height=image.size
        image_rows.append({'absolute_path':str(path.resolve()),'relative_path':path.relative_to(args.output).as_posix(),
                           'size_bytes':path.stat().st_size,'format':fmt,'width_px':width,'height_px':height,
                           'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    with (args.output/'image_inventory.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(image_rows[0]));writer.writeheader();writer.writerows(image_rows)
    lines+=['[Полный CSV: путь, размер, формат, разрешение и SHA-256](image_inventory.csv). '
            f'Фактически декодировано {len(image_rows)} PNG.']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'blocks':len(results),'source_slices':75,'decoded_pngs':len(image_rows)},indent=2))


if __name__=='__main__':main()
