# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs
from pathlib import Path

block_cipher = None

# Collect dependencies
datas = [
    ('gui/web', 'gui/web'),
    ('voices', 'voices'),
]
binaries = []
hiddenimports = [
    'sounddevice',
    '_cffi_backend',
    'scipy',
    'scipy.signal',
    'scipy.ndimage',
    'numpy',
    'pydantic',
    'pydantic_core',
    'onnxruntime',
    'core',
    'core.audio_io',
    'core.ring_buffer',
    'core.noise_gate',
    'core.crossfade',
    'core.audio_quality',
    'core.pipeline',
    'core.worker_pipeline',
    'models',
    'models.voice_manager',
    'models.inference_engine',
    'models.rvc_pipeline',
    'gui',
    'gui.server',
]

# Collect onnxruntime binaries and data
ort_datas, ort_binaries, ort_hidden = collect_all('onnxruntime')
datas += ort_datas
binaries += ort_binaries
hiddenimports += ort_hidden

# Collect sounddevice binaries
sd_datas, sd_binaries, sd_hidden = collect_all('sounddevice')
datas += sd_datas
binaries += sd_binaries
hiddenimports += sd_hidden

# Collect pywebview and .NET bindings for native Windows window
wv_datas, wv_binaries, wv_hidden = collect_all('webview')
datas += wv_datas
binaries += wv_binaries
hiddenimports += wv_hidden

pynet_datas, pynet_binaries, pynet_hidden = collect_all('pythonnet')
datas += pynet_datas
binaries += pynet_binaries
hiddenimports += pynet_hidden

clr_datas, clr_binaries, clr_hidden = collect_all('clr_loader')
datas += clr_datas
binaries += clr_binaries
hiddenimports += clr_hidden

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'notebook', 'IPython'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Tvoice-s',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Tvoice-s',
)
