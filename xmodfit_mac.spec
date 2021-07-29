# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None
homedir='/Users/mrinalkb/PycharmProjects/XModFit'

hutils=collect_submodules('utils')
hchemform=collect_submodules('Chemical_Formula')
hnumba=collect_submodules('numba')
hnumba_scipy=collect_submodules('numba_scipy')
hllvmlite=collect_submodules('llvmlite')
hnnumpy=collect_submodules('numba-numpy')
hstatsmodels=collect_submodules('statsmodels')
hpatsy=collect_submodules('patsy')

all_hidden_imports=hutils+hchemform+hnumba+hnumba_scipy+hllvmlite+hstatsmodels+hpatsy

a = Analysis(['xmodfit.py'],
             pathex=[homedir],
             binaries=[('/Users/mrinalkb/opt/anaconda3/lib/python3.8/site-packages/llvmlite/binding/libllvmlite.dylib','.'),],
             datas=[(homedir+'/Fortran_routines','Fortran_routines'),
             (homedir+'/Function_Details','Function_Details'),
             (homedir+'/Tools','Tools'),
             (homedir+'/UI_Forms','UI_Forms'),
             (homedir+'/Functions','Functions'),
             (homedir+'/functionTemplate.txt','.'),
             (homedir+'/license','.'),
             (homedir+'/xraydb','xraydb'),
             (homedir+'/mendeleev','mendeleev')],
             hiddenimports=all_hidden_imports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='xmodfit',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='xmodfit')
