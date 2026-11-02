#!/usr/bin/env python3
"""Prepare a separate runtime adapter for the unchanged C++ detectors.

Only paths, slice-list positions and diagnostic saving are adapted. The
original source is read, never overwritten. Build the generated CMake project,
then pass its binary to this script to run named real source slices.
"""
import argparse
import csv
import difflib
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess


def prepare(source, output):
    original=source.read_text()
    changed=original
    edits=[(r'static const std::string INPUT_FOLDER =.*?;', 'static std::string INPUT_FOLDER;'),
           (r'static const std::string OUTPUT_FOLDER =.*?;', 'static std::string OUTPUT_FOLDER;')]
    edits += [(r'constexpr int '+key+r'\s*=\s*\d+;', 'static int '+key+';')
              for key in ['FIRST_SLICE','CENTER_SLICE','LAST_SLICE']]
    edits += [(r'constexpr bool SAVE_ALL_SLICE_DIAGNOSTICS = true;',
               'constexpr bool SAVE_ALL_SLICE_DIAGNOSTICS = false;')]
    start=('int main(int argc, char** argv)\n{\n'
           '    if (argc != 6) { std::cerr << "expected input output first center last\\n"; return 2; }\n'
           '    INPUT_FOLDER = argv[1];\n    OUTPUT_FOLDER = argv[2];\n'
           '    FIRST_SLICE = std::stoi(argv[3]);\n    CENTER_SLICE = std::stoi(argv[4]);\n'
           '    LAST_SLICE = std::stoi(argv[5]);')
    edits += [(r'int main\(\)\n\{',start)]
    for pattern,replacement in edits:
        changed,n=re.subn(pattern,lambda _:replacement,changed,count=1,flags=re.S)
        if n!=1:raise ValueError('Unsupported source layout: '+pattern)
    output.mkdir(parents=True,exist_ok=True)
    (output/'baseline_runtime.cpp').write_text(changed)
    (output/'adapter.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True),changed.splitlines(True),fromfile='original/main.cpp',tofile='baseline_runtime.cpp')))
    (output/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.16)\n'
        'project(ct_baseline LANGUAGES CXX)\nset(CMAKE_CXX_STANDARD 17)\n'
        'find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)\n'
        'add_executable(baseline_runtime baseline_runtime.cpp)\n'
        'target_include_directories(baseline_runtime PRIVATE ${OpenCV_INCLUDE_DIRS})\n'
        'target_link_libraries(baseline_runtime PRIVATE ${OpenCV_LIBS})\n')
    provenance={'original_source':str(source.resolve()),
        'original_sha256':hashlib.sha256(original.encode()).hexdigest(),
        'adapted_sha256':hashlib.sha256(changed.encode()).hexdigest(),
        'detector_algorithms_and_parameters_changed':False,
        'adapter_changes':['runtime input/output paths','runtime sorted-file list indices',
                           'SAVE_ALL_SLICE_DIAGNOSTICS=false']}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    return provenance


def run(binary, source, output, centers, half):
    import numpy as np
    from PIL import Image
    files=sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in ('.tif','.tiff'))
    indices={}
    for i,p in enumerate(files):
        m=re.fullmatch(r'img_(\d+)\.tiff?',p.name,re.I)
        if not m:raise ValueError('Unrecognised TIFF source name: '+p.name)
        indices[int(m.group(1))]=i
    records=[]
    for center in centers:
        ci=indices[center];first=max(0,ci-half);last=min(len(files)-1,ci+half)
        rows=[{'ordinal':i,'source_file':files[i].name,'source_slice':next(n for n,j in indices.items() if j==i)}
              for i in range(first,last+1)]
        if any(rows[k+1]['source_slice']-rows[k]['source_slice']!=1 for k in range(len(rows)-1)):
            raise ValueError('Nonconsecutive real source slices')
        out=output/str(center)
        if out.exists() and any(out.iterdir()):raise ValueError('Use a new output directory: '+str(out))
        out.mkdir(parents=True,exist_ok=True)
        with (out/'source_mapping.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        with (out/'run.log').open('w') as log:
            subprocess.run([str(binary.resolve()),str(source.resolve()),str(out.resolve()),
                            str(first),str(ci),str(last)],stdout=log,stderr=subprocess.STDOUT,check=True)
        report=(out/'report_three_detectors.txt').read_text()
        marker='slice,standard_count,adaptive_count,pc_count,'
        table=list(csv.DictReader(io.StringIO(report[report.index(marker):])))
        center_row=next(r for r in table if int(r['slice'])==ci)
        record={'center_source_slice':center,'center_ordinal':ci,
                'source_slices':[r['source_slice'] for r in rows],
                'usable_for_3d_canny_comparison':first<ci<last,'counts':{}}
        for method,folder,prefix,key in [
            ('standard','03_canny_standard','standard','standard_count'),
            ('adaptive','04_canny_adaptive','adaptive','adaptive_count'),
            ('phase','07_phase_congruency_edges','pc_edges','pc_count')]:
            path=out/folder/f'{prefix}_{ci:04d}.png'
            with Image.open(path) as im:a=np.asarray(im)
            with Image.open(files[ci]) as im:expected=np.asarray(im).shape
            assert a.shape==expected and a.dtype==np.uint8 and set(np.unique(a))<={0,255}
            assert int(np.count_nonzero(a))==int(center_row[key])
            shutil.copyfile(path,out/f'{method}_{center:04d}.png')
            record['counts'][method]=int(np.count_nonzero(a))
        (out/'validation.json').write_text(json.dumps(record,indent=2))
        records.append(record)
        print(json.dumps(record),flush=True)
    (output/'validation_all.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='Original main.cpp')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--binary',type=Path)
    p.add_argument('--input',type=Path)
    p.add_argument('--centers',default='625,750,875,988')
    p.add_argument('--half',type=int,default=8)
    a=p.parse_args()
    prepare(a.source,a.out)
    if a.binary:
        if not a.input:p.error('--binary requires --input')
        run(a.binary,a.input,a.out,list(map(int,a.centers.split(','))),a.half)
