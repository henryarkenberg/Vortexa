# Packs a ready-to-use Vortexa release into a zip file.
# Run from the project root:  powershell scripts\package-release.ps1

$ErrorActionPreference = "Stop"

cargo build --release

$Exe = "target\release\vortexa.exe"
if (-not (Test-Path $Exe)) {
    Write-Error "Build failed: $Exe not found"
    exit 1
}

$Version = (Select-String -Path "Cargo.toml" -Pattern '^version\s*=\s*"(.*)"').Matches[0].Groups[1].Value
$Dir = "dist\vortexa-release"
$Zip = "dist\vortexa-$Version-windows-x64.zip"

Remove-Item -Recurse -Force $Dir, $Zip -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $Dir | Out-Null
New-Item -ItemType Directory -Force "$Dir\data" | Out-Null

Copy-Item $Exe "$Dir\vortexa.exe"
Copy-Item "README.md" $Dir
Copy-Item "LICENSE" $Dir
Copy-Item "data\.gitkeep" "$Dir\data\.gitkeep"

Compress-Archive -Path "$Dir\*" -DestinationPath $Zip
Write-Host "Packed: $Zip"
