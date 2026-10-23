$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:TEMP = Join-Path $PSScriptRoot 'tmp'
$env:TMP = $env:TEMP
$env:PYTHONIOENCODING = 'utf-8'
$env:YOLO_CONFIG_DIR = Join-Path $PSScriptRoot 'tmp\ultralytics'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot 'tmp\matplotlib'
$env:TORCH_HOME = Join-Path $PSScriptRoot 'tmp\torch'
$env:XDG_CACHE_HOME = Join-Path $PSScriptRoot 'tmp\cache'
$env:HF_HOME = Join-Path $PSScriptRoot 'tmp\huggingface'
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot 'tmp\uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot 'tmp\uv-python'
New-Item -ItemType Directory -Force -Path $env:TEMP, $env:YOLO_CONFIG_DIR, $env:MPLCONFIGDIR | Out-Null
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'main.py') @args
exit $LASTEXITCODE
