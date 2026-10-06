# Install (or reinstall) the latest released Sauron.exe and make it start
# with Windows. After this, Sauron keeps itself up to date.
#
#   powershell -ExecutionPolicy Bypass -File install.ps1           # install + launch
#   powershell -ExecutionPolicy Bypass -File install.ps1 -NoLaunch
param([switch]$NoLaunch)
$ErrorActionPreference = "Stop"

$repo       = "facubara/sauron"
$installDir = Join-Path $env:LOCALAPPDATA "Programs\Sauron"
$exe        = Join-Path $installDir "Sauron.exe"
$shortcut   = Join-Path ([Environment]::GetFolderPath("Startup")) "Sauron.lnk"

$release = Invoke-RestMethod "https://api.github.com/repos/$repo/releases/latest" `
    -Headers @{ "User-Agent" = "Sauron-installer" }
$asset = $release.assets | Where-Object name -eq "Sauron.exe" | Select-Object -First 1
if (-not $asset) { throw "Release $($release.tag_name) has no Sauron.exe asset" }

# Stop running copies (any location) so the exe can be replaced.
Get-Process Sauron -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

New-Item -ItemType Directory -Force $installDir | Out-Null
$tmp = "$exe.download"
Write-Host "Downloading $($release.tag_name) ($([math]::Round($asset.size / 1MB)) MB)..."
Invoke-WebRequest $asset.browser_download_url -OutFile $tmp -UseBasicParsing

if ($asset.digest -and $asset.digest.StartsWith("sha256:")) {
    $actual = (Get-FileHash $tmp -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $asset.digest.Substring(7)) { Remove-Item $tmp; throw "SHA-256 mismatch" }
}
Move-Item -Force $tmp $exe
Remove-Item -ErrorAction SilentlyContinue "$exe.old", "$exe.new"

$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut($shortcut)
$lnk.TargetPath = $exe
$lnk.WorkingDirectory = $installDir
$lnk.IconLocation = "$exe,0"
$lnk.Save()

Write-Host "Installed $($release.tag_name) to $exe"
Write-Host "Startup shortcut: $shortcut"
if (-not $NoLaunch) { Start-Process $exe -WorkingDirectory $installDir }
