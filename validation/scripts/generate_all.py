"""Regenerate GRIM evidence, full specifications and compact workbook examples."""
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
temp=ROOT/'validation'/'.work'/'tmp'
temp.mkdir(parents=True,exist_ok=True)
os.environ.update(TEMP=str(temp),TMP=str(temp),QT_QPA_PLATFORM='offscreen')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'2')

if __name__=='__main__':
    import arithmetic_cases,coordinate_cases,physics_cases,plot_cases,compare_cases
    for module in (arithmetic_cases,coordinate_cases,physics_cases,plot_cases,compare_cases):
        module.generate()
        print(module.__name__+' complete')
    import compact_arithmetic,compact_coordinates,compact_physics,compact_plots
    for module in (compact_arithmetic,compact_coordinates,compact_physics,compact_plots):
        module.generate()
        print(module.__name__+' complete')
