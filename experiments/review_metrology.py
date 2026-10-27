"""Copy verified CT study outputs into Git and create the review report."""
import argparse,csv,hashlib,json,re,shutil,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from boundary_study import plt,load
from compare_metrology import geometry
from canny3d_study import plane_segments
from matplotlib.collections import LineCollection


def read_ply(path):
    with path.open('rb') as stream:
        lines=[]
        while True:
            line=stream.readline().decode().strip();lines.append(line)
            if line=='end_header':break
        nv=int(next(x.split()[-1] for x in lines if x.startswith('element vertex ')))
        nf=int(next(x.split()[-1] for x in lines if x.startswith('element face ')))
        points=np.frombuffer(stream.read(nv*12),dtype='<f4').reshape(nv,3)
        records=np.frombuffer(stream.read(),dtype=[('n','u1'),('indices','<i4',(3,))])
    assert len(records)==nf and (records['n']==3).all()
    return points,records['indices'],lines


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--farid',type=Path,required=True)
    p.add_argument('--phantom',type=Path,required=True)
    p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--refresh-report',action='store_true',help='Recheck an existing export and regenerate only reports/plots')
    args=p.parse_args();repo=Path.cwd().resolve();output=args.output.resolve()
    # Completion is required, rather than exporting a directory mid-write.
    metrics=json.loads((args.comparison/'metrics.json').read_text())
    assert len(metrics)==5
    if not args.refresh_report:
        shutil.copytree(args.comparison,output)
        shutil.copytree(args.farid,output/'farid3d')
        shutil.copytree(args.phantom,output/'phantoms')
        shutil.copytree(args.pilot,output/'diagnostics')
    else:assert output.is_dir()
    checks=[]
    for path in sorted(output.rglob('surface_coordinates.npz')):
        data=np.load(path)
        base,moved,faces=data['seed_zyx'],data['refined_zyx'],data['faces_zyx']
        assert len(base)==len(moved)==len(data['accepted'])
        assert faces.min()>=0 and faces.max()<len(moved)
        assert np.linalg.norm(moved-base,axis=1).max()<2.401
        check=geometry(base,moved,faces)
        meta=json.loads(path.with_name('metrics.json').read_text())
        assert check==meta['geometry']
        if path.parent.name=='farid3d':first=738;y0,x0=459,286
        else:
            first=meta['first'];y0,_,x0,_=meta['crop_box_y0_y1_x0_x1']
        expected=moved[:,[2,1,0]].copy();expected[:,:2]+=[x0,y0]
        ply=path.with_name('surface_voxels.ply')
        if check['passed']:
            points,actual_faces,header=read_ply(ply)
            assert np.allclose(points,expected,atol=1e-4)
            assert np.array_equal(actual_faces,faces[:,[0,2,1]])
            physical=path.with_name('surface_mm.ply')
            if physical.exists():
                xyz,pfaces,header=read_ply(physical)
                expected[:,2]+=first-500;expected*=.0754
                assert np.allclose(xyz,expected,atol=1e-5)
                assert any('units=mm' in line for line in header)
                assert np.array_equal(pfaces,actual_faces)
        else:assert not ply.exists()
        checks.append({'file':str(path.relative_to(repo)),**check})
    (output/'export_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    transition=output/'block_0738_0762'
    a=np.load(transition/'local_rsf_6.npz')['mask'];b=np.load(transition/'local_rsf_12.npz')['mask']
    raw=load(Path('CT_Test'),750)[459:1370,286:697]
    fig,axes=plt.subplots(1,2,figsize=(10,10))
    for ax,title in zip(axes,['Original CT','Red = disagreement between scales 6 and 12']):
        ax.imshow(raw,cmap='gray',vmin=.06,vmax=.8);ax.axis('off');ax.set_title(title,fontsize=9)
    overlay=np.zeros((*raw.shape,4));overlay[a[12]!=b[12]]=[1,.15,.1,.85]
    axes[1].imshow(overlay);fig.tight_layout()
    fig.savefig(transition/'scale_sensitivity_0750.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,9))
    for ax in axes:
        ax.imshow(raw,cmap='gray',vmin=.06,vmax=.8);ax.axis('off')
        ax.set_xlim(110,360);ax.set_ylim(480,50)
    axes[0].set_title('Original CT')
    for ax,key,title in zip(axes[1:],['preserved_topology','local_rsf_6'],
            ['Preserved topology + PSF','Local regions + PSF']):
        data=np.load(transition/key/'surface_coordinates.npz')
        v,f,a=data['refined_zyx'],data['faces_zyx'],data['accepted']
        ok=a[f].all(axis=1)
        for selected,color in [(ok,'#00ff70'),(~ok,'#ff9d00')]:
            segments=plane_segments(v,f[selected],12.)
            ax.add_collection(LineCollection(segments[:,:,[1,0]],colors=color,linewidths=.7))
        ax.set_title(title+'\ngreen=fit passed; orange=unconfirmed',fontsize=10)
    fig.suptitle('img_0750: both meshes pass geometry checks; real CT accuracy is not yet measured',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(transition/'quality_validated_0750.png',dpi=160);plt.close(fig)
    benchmark=json.loads((output/'phantoms/phantom_benchmark.json').read_text())
    lines=['# Сравнение методов определения поверхностей КТ — 28 октября 2026','',
        '**Пропуски границ на переходном сечении полностью не устранены.** '
        'Лучшее улучшение локализации получено модельной подгонкой профиля на фантоме. '
        'На реальной КТ некоторые профили и стенки остаются неоднозначными; '
        'эти результаты пока нельзя считать метрологически подтверждённой границей.','',
        '[Поиск литературы, проверенные источники и ограничения доступа](../../experiments/CT_SURFACE_LITERATURE.md). '
        'Прямой поиск/полные тексты Crossref, NIST, IEEE, Springer, arXiv и PLOS '
        'блокируются шлюзом HTTP 403; использованы доступные открытые реализации и их библиография.','',
        '## Посмотреть сравнения','',
        '- [Переходное сечение 750: прежний Canny, PSF, локальные области и предложения стенок](block_0738_0762/comparison_0750.png)',
        '- [Слабые стенки крупнее: зелёный — профиль принят, оранжевый — не подтверждён](block_0738_0762/quality_validated_0750.png)',
        '- [Производные Farid–Simoncelli против прежних гауссовых](farid3d/farid_comparison_0750.png)',
        '- [Чувствительность к размеру локальной области — это не карта ошибки](block_0738_0762/scale_sensitivity_0750.png)',
        '- [Толстое сечение 512](block_0500_0524/comparison_0512.png)',
        '- [Тонкое сечение 988](block_0976_1000/comparison_0988.png)','',
        'Зелёная линия в обычном сравнении показывает всю оценённую поверхность. '
        'Рисунки `quality_validated` и `weak_walls` отдельно кодируют качество профиля. '
        'Прохождение критерия подгонки не гарантирует отсутствие систематической ошибки КТ.','',
        '**Правый столбец `profile_wall_proposals` в общем сравнении — диагностический:** '
        'его сетка имеет 33 неманифолдных ребра и не распространяется в PLY. '
        'Рекомендуемое крупное изображение `quality_validated_0750` показывает два '
        'варианта, прошедших геометрическую проверку.','',
        '## Что применено и что получилось','',
        '| Метод | Результат и ограничение |','|---|---|',
        '| 3D Farid–Simoncelli + адаптивный Canny | Сравнён на тех же исходных масках. Пропуски стенок сохраняются; на фантоме выигрыша относительно Gaussian не получено. |',
        '| Локальная region-scalable energy + 3D TV, масштабы 6 и 12 | Может менять связность. Некоторые верхние границы добавились, но часть слабых стенок по-прежнему теряется; заметна зависимость от масштаба. |',
        '| Начальная разметка по локальному контрасту + локальная энергия, масштабы 3 и 6 | Даёт больше кандидатов, но также ложные структуры в каналах и материале. Не принята как итоговая сегментация. |',
        '| Гессиан + подгонка двух размытых поверхностей | Консервативные предложения новых тонких стенок по измеренному контрасту. Не восстанавливают все пропуски; остаются экспериментальными. |',
        '| Подгонка PSF-профиля исходной КТ вдоль 3D-нормали | Лучший результат локализации на фантоме; на реальной КТ ограничивает уточнение по качеству модели. Сохраняющий топологию вариант не создаёт пропущенные каналы. |','',
        'В отличие от сглаживания готовой сетки, PSF-подгонка использует интенсивность исходной КТ. '
        'Проверяются одиночный скачок и пара поверхностей тонкой стенки, линейный фон, '
        'остаток и идентифицируемость. Неподтверждённые точки остаются исходными; '
        'опасные для граней сдвиги уменьшаются и отмечаются как неподтверждённые.','',
        '## Проверка координат на известной геометрии','',
        'Одинаковые начальные маски сферы с наклонным каналом, неоднородной яркостью '
        'и тремя независимыми добавками шума. В таблице ошибка относительно аналитических поверхностей; '
        'генератор сначала вокселизует геометрию и размывает её. Значения относятся только к этому фантому.','',
        '| Метод | P95 ошибки, vox, диапазон трёх опытов |','|---|---:|']
    for name,label in [('gradient','Gaussian + 3D Canny/градиент'),('farid','Farid + 3D Canny/градиент'),('psf','Модель профиля PSF')]:
        values=[x['errors_against_analytic_surfaces']['refined']['p95_vox'] for x in benchmark['volume_cases'] if x['method']==name]
        lines.append(f'| {label} | {min(values):.3f}–{max(values):.3f} |')
    thin=benchmark['thin_wall_profiles']
    lines+=['',f"На 80 профилях тонких стенок: P95 модели одиночной границы = {thin['single_interface']['p95_vox']:.3f} vox, "
        f"двух поверхностей = {thin['paired_interfaces']['p95_vox']:.3f} vox. "
        'При очень тонкой стенке амплитуда, ширина и PSF связаны; неоднозначные пары не принимаются автоматически.','',
        '- [Поверхности фантома](phantoms/phantom_surfaces.png)',
        '- [Профиль тонкой стенки](phantoms/thin_wall_profiles.png)',
        '- [Все численные результаты](phantoms/phantom_benchmark.json)','',
        '## На реальных данных: качество подгонки, не точность','',
        '**75 срезов** в блоках 500–524, 738–762, 976–1000 плюс реальный контекст до восьми срезов. '
        'Фильтры не получают искусственно повторённых TIFF. Маска в контексте — '
        'начальная геометрическая оценка, её происхождение описано в коде.','',
        '| Блок | Вариант | Вершины с принятой подгонкой | P95 смещения, vox | Геометрия сетки |',
        '|---|---|---:|---:|---|']
    for m in metrics:
        lines.append(f"| {m['first']}–{m['last']} | {m['method']} | {m['accepted_vertex_fraction']*100:.1f}% | "
            f"{m['abs_shift_p95_vox']:.3f} | {'прошла' if m['geometry']['passed'] else 'отклонена'} |")
    lines+=['','Проценты выше — прохождение выбранных критериев модели; их нельзя сравнивать '
        'с долей рёбер Canny как оценки точности. Смещение поверхности тоже не равно ошибке.','',
        '## Геометрия, файлы и единицы','',
        'Сохранены PNG, численные маски/координаты NPZ и пересечения с исходными срезами CSV.GZ. '
        'PLY распространяются только для моделей, прошедших проверку конечности координат, '
        'индексов, вырождения/переворота граней, манифолдности рёбер и открытых краёв только на торцах. '
        'Глобальная проверка пересечений удалённых треугольников не выполнена. '
        'Диагностические NPZ могут содержать отклонённую сетку; её статус есть в metrics.json.','',
        '`surface_voxels.ply`: X/Y в координатах исходного изображения, Z=0 у первого среза блока. '
        '`surface_mm.ply`: миллиметры, X/Y от пикселя (0,0), Z=0 на img_0500. '
        'Кубический воксель 75,4 мкм подтверждён пользователем; независимая калибровка сканера здесь не проверялась. '
        'Фрагменты открыты на торцах и не являются полной замкнутой моделью детали.','',
        'Скачать фрагменты в миллиметрах (на странице GitHub — Download raw file):','',
        '- [500–524: сохранённая топология + PSF](block_0500_0524/preserved_topology/surface_mm.ply)',
        '- [738–762: сохранённая топология + PSF](block_0738_0762/preserved_topology/surface_mm.ply)',
        '- [738–762: локальные области + PSF](block_0738_0762/local_rsf_6/surface_mm.ply)',
        '- [976–1000: сохранённая топология + PSF](block_0976_1000/preserved_topology/surface_mm.ply)','',
        '[Проверка фактически прочитанных NPZ/PLY](export_validation.json). '
        '[Полный перечень PNG: абсолютные пути, байты, формат, размер и SHA-256](image_inventory.csv).','',
        '## Что необходимо для проверки требуемой точности','',
        '`lopatka.stl` не найден в доступных файлах и Git-истории. Без него или иной '
        'контрольной геометрии точность на реальной детали не измерена. После получения модели '
        'нужны проверка единиц, жёсткое совмещение, отдельная оценка наружных и внутренних '
        'поверхностей с P95/максимумом расхождения. Ручная модель сама может иметь погрешность. '
        'Воксельный шаг известен, но требуемый допуск пока не задан.','',
        '## Воспроизведение','',
        'Из корня репозитория с зависимостями experiments/requirements.txt. Каталоги вывода должны быть новыми.','',
        '```bash',
        'python -m unittest discover -s experiments -p "test_*.py" -v',
        'python experiments/metrology_study.py --output /tmp/ct-local-regions --rounds 4 --scales 6,12',
        'python experiments/compare_metrology.py --regions /tmp/ct-local-regions --output /tmp/ct-model-comparison',
        'python experiments/compare_farid3d.py --output /tmp/ct-farid3d',
        'python experiments/benchmark_metrology.py --output /tmp/ct-phantoms',
        '```','',
        'В diagnostics сохранены также промежуточные отвергнутые визуальные опыты: локальные '
        'пороги инициализации, RSF с другой начальной маской, объединение масок и '
        'сохранение только крупнейшей компоненты. Последнее может удалять реальные стенки '
        'в коротком блоке и не используется как физическое подтверждение поверхности. '
        'Они не являются выбранным автоматическим решением.','',
        'Исходные CT_Test, main.cpp и CMake не изменены. Все исходные изображения '
        'экспериментов в рабочих каталогах сохранены; копии для GitHub сверены по SHA-256.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    provenance={'base_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'voxel_size_um':[75.4]*3,'voxel_size_source':'confirmed by user in this conversation',
        'reference_stl_found':False,'real_CT_accuracy_measured':False,'sources':{}}
    for source in sorted(Path('experiments').glob('*.py')):
        provenance['sources'][str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    inputs=[]
    for first,last in [(500,532),(730,770),(968,1000)]:
        for index in range(first,last+1):
            source=Path('CT_Test')/f'img_{index:04d}.tif'
            with Image.open(source) as im:
                assert im.size==(910,1804) and im.mode=='I;16'
            inputs.append({'path':str(source),'size_bytes':source.stat().st_size,
                           'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    (output/'input_manifest.json').write_text(json.dumps(inputs,indent=2)+'\n')
    origins={}
    for source,target in [(args.comparison,output),(args.farid,output/'farid3d'),
                          (args.phantom,output/'phantoms'),(args.pilot,output/'diagnostics')]:
        for image in source.rglob('*.png'):origins[target/image.relative_to(source)]=image.resolve()
    inventory=[]
    for path in sorted(output.rglob('*.png')):
        with Image.open(path) as im:
            im.load();fmt,width,height,mode=im.format,*im.size,im.mode
        sha=hashlib.sha256(path.read_bytes()).hexdigest();original=origins.get(path,path)
        assert path.stat().st_size>0 and sha==hashlib.sha256(original.read_bytes()).hexdigest()
        inventory.append({'original_absolute_path':str(original),'repository_absolute_path':str(path),
            'repository_path':str(path.relative_to(repo)),'size_bytes':path.stat().st_size,
            'format':fmt,'width':width,'height':height,'mode':mode,'sha256':sha})
    with (output/'image_inventory.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(inventory[0]),lineterminator='\n');writer.writeheader();writer.writerows(inventory)
    (output/'image_verification.json').write_text(json.dumps({'decoded_images':len(inventory),
        'total_image_bytes':sum(x['size_bytes'] for x in inventory),'all_source_copies_sha256_equal':True},indent=2)+'\n')
    print('Exported and verified',len(inventory),'images; checked',len(checks),'numeric models',flush=True)


if __name__=='__main__':main()
