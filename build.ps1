# Build dist\Sauron.exe. Run from the repo root:  .\build.ps1
# Requires:  pip install -r requirements-dev.txt
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "== Lint"  -ForegroundColor Cyan
python -m ruff check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== Tests" -ForegroundColor Cyan
python -m pytest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== PyInstaller" -ForegroundColor Cyan
python -m PyInstaller --noconfirm --clean Sauron.spec
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$exe = Join-Path $PSScriptRoot "dist\Sauron.exe"
$size = [math]::Round((Get-Item $exe).Length / 1MB, 1)
Write-Host "== Built $exe ($size MB)" -ForegroundColor Green
