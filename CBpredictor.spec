# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['desktop_app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('db', 'db'),
        ('analysis', 'analysis'),
        ('models', 'models'),
        ('config.py', '.'),
    ],
    hiddenimports=[
        'akshare', 'numpy', 'sqlalchemy', 'sqlite3',
        'db.models', 'analysis.ml_model_v5', 'analysis.fundamental_features',
        'analysis.model_persistence'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, block_cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CBpredictor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codegen arch=None,
    init_script=None,
    upx_exe=None,
    upx_arch_filter=None,
    as_package='',
    runtime_tmpdir=None,
    console_window_title='可转债预测工具',
    target_name='CBpredictor',
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CBpredictor',
)