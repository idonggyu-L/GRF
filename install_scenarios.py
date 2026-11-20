"""Copy the scenarios in ./scenarios into the installed gfootball package.

GRF loads a scenario by name from the `gfootball.scenarios` package (each file does
`from . import *`), so the files have to live inside that package directory.
Identical files are left alone; pass --force to overwrite files that differ.
"""
import argparse
import filecmp
import glob
import os
import shutil

import gfootball.scenarios as scenarios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='overwrite scenario files that differ')
    args = ap.parse_args()
    dst = os.path.dirname(scenarios.__file__)
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scenarios')
    for f in sorted(glob.glob(os.path.join(src, '*.py'))):
        out = os.path.join(dst, os.path.basename(f))
        if os.path.exists(out) and filecmp.cmp(f, out, shallow=False):
            print(f'same      {out}')
        elif os.path.exists(out) and not args.force:
            print(f'DIFFERS   {out} (kept; rerun with --force to overwrite)')
        else:
            shutil.copy2(f, out)
            print(f'installed {out}')


if __name__ == '__main__':
    main()
