# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
<#
.SYNOPSIS
  Install the Windows build of habitat-sim and test it on this PC's GPU.

.DESCRIPTION
  1. Checks for Python 3.10 (64-bit), Git (with Git LFS) and an NVIDIA driver.
  2. Creates a virtual environment in the test folder and installs the latest
     habitat-sim Windows wheel from the windows-wheel-latest release.
  3. Downloads the Habitat test scenes and the Habitat 3.0 humanoids.
  4. Runs the smoke test (datasets, RGB/depth/semantic rendering, physics, FPS)
     and the humanoid test (SMPL-X avatar, skinning, human mask, FPS).
  5. With -WithHabitatLab: installs Habitat-Lab with the Windows patch and runs
     the Habitat 3.0 humanoid episode + benchmark (downloads ~1-2 GB more).
  Everything is written under the test folder; results are zipped at the end.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File gpu_test.ps1
  powershell -ExecutionPolicy Bypass -File gpu_test.ps1 -WithHabitatLab
#>
param(
    [string]$Dir = "$HOME\habitat-test",
    [switch]$WithHabitatLab,
    [string]$Repo = "Achilles1995/habitat-sim",
    [string]$Branch = "windows-port"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # much faster Invoke-WebRequest
$HabitatLabSha = "0fb6f43ffe806a8088a171b036336c093bcf604e"

New-Item -ItemType Directory -Force -Path $Dir | Out-Null
$Results = Join-Path $Dir "results"
New-Item -ItemType Directory -Force -Path $Results | Out-Null
$Log = Join-Path $Results "gpu_test_log.txt"
Start-Transcript -Path $Log -Force | Out-Null

function Step($msg) { Write-Host ""; Write-Host "==== $msg ====" -ForegroundColor Cyan }
function Fail($msg) {
    Write-Host "ERROR: $msg" -ForegroundColor Red
    Stop-Transcript | Out-Null
    exit 1
}
# Run python with output shown and saved to a log. Windows PowerShell 5.1 turns
# redirected stderr lines into errors, which with -ErrorAction Stop would abort
# the script on harmless warnings, so relax that around the call.
function RunLogged([string]$logFile, [string[]]$argList) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $script:vpy @argList 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $logFile | Out-Host
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $old }
    return $code
}
function Run($exe, [string[]]$argList) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $exe @argList 2>&1 | ForEach-Object { "$_" } | Out-Host; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($code -ne 0) { throw "'$exe $($argList -join ' ')' failed with exit code $code" }
}

$summary = [ordered]@{}

# ---------------------------------------------------------------- 1. checks
Step "1/5 Checking prerequisites"
$py = $null
try { $v = & py -3.10 -c "import sys,struct; print(sys.version.split()[0], struct.calcsize('P')*8)"; if ($LASTEXITCODE -eq 0) { $py = "py" } } catch {}
if (-not $py) {
    Fail "Python 3.10 (64-bit) not found. Install it from https://www.python.org/downloads/release/python-31011/ (Windows installer 64-bit), then run this script again."
}
Write-Host "Python: $v"
if ($v -notmatch " 64$") { Fail "Python 3.10 is 32-bit; please install the 64-bit version." }

try { $gitv = & git --version } catch { Fail "Git not found. Install Git for Windows from https://git-scm.com/download/win and run again." }
Write-Host "Git: $gitv"
try { & git lfs version | Out-Host } catch { Fail "Git LFS not found. Reinstall Git for Windows with the 'Git LFS' option ticked." }
& git lfs install --skip-repo | Out-Null

try {
    $gpu = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    Write-Host "NVIDIA GPU: $gpu"
    $summary["GPU (nvidia-smi)"] = "$gpu"
} catch {
    Write-Host "WARNING: nvidia-smi not found - is the NVIDIA driver installed?" -ForegroundColor Yellow
    $summary["GPU (nvidia-smi)"] = "nvidia-smi not found"
}

# ---------------------------------------------------------------- 2. install
Step "2/5 Creating virtual environment and installing habitat-sim"
$venv = Join-Path $Dir "venv"
if (-not (Test-Path "$venv\Scripts\python.exe")) { Run "py" @("-3.10", "-m", "venv", $venv) }
$script:vpy = "$venv\Scripts\python.exe"
$vpy = $script:vpy
Run $vpy @("-m", "pip", "install", "-q", "-U", "pip")

# Prefer the NVIDIA GPU for this Python (matters on laptops with Intel/AMD + NVIDIA graphics).
# Same as Settings > System > Display > Graphics > python.exe > High performance; per-user, only for this venv.
try {
    $key = "HKCU:\Software\Microsoft\DirectX\UserGpuPreferences"
    if (-not (Test-Path $key)) { New-Item -Path $key -Force | Out-Null }
    $realpy = & $vpy -c "import sys; print(sys._base_executable)"
    foreach ($exe in @((Resolve-Path $vpy).Path, $realpy)) {
        New-ItemProperty -Path $key -Name $exe -Value "GpuPreference=2;" -PropertyType String -Force | Out-Null
    }
    Write-Host "Set 'High performance' GPU preference for: $realpy"
} catch { Write-Host "Could not set GPU preference (not critical): $_" -ForegroundColor Yellow }

$rel = Invoke-RestMethod -UseBasicParsing "https://api.github.com/repos/$Repo/releases/tags/windows-wheel-latest"
$wheelAsset = $rel.assets | Where-Object { $_.name -like "*-cp310-cp310-win_amd64.whl" } | Select-Object -First 1
$infoAsset = $rel.assets | Where-Object { $_.name -eq "BUILD_INFO.txt" } | Select-Object -First 1
if (-not $wheelAsset) { Fail "No wheel found in the windows-wheel-latest release of $Repo." }
$wheelPath = Join-Path $Dir $wheelAsset.name
Write-Host "Downloading $($wheelAsset.name) ($([math]::Round($wheelAsset.size/1MB,1)) MB)"
Invoke-WebRequest -UseBasicParsing $wheelAsset.browser_download_url -OutFile $wheelPath
if ($infoAsset) { Invoke-WebRequest -UseBasicParsing $infoAsset.browser_download_url -OutFile (Join-Path $Results "BUILD_INFO.txt"); Get-Content (Join-Path $Results "BUILD_INFO.txt") | Out-Host }

if ($WithHabitatLab) {
    # habitat-lab pins numpy 1.26.4
    Set-Content -Path (Join-Path $Dir "constraints.txt") -Value "numpy==1.26.4"
    Run $vpy @("-m", "pip", "install", "-q", "-c", (Join-Path $Dir "constraints.txt"), "numpy", "attrs", "gitpython", "imageio", "imageio-ffmpeg", "matplotlib", "numba", "numpy-quaternion", "pillow==10.4.0", "scipy", "tqdm", "psutil")
    Run $vpy @("-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", $wheelPath)
} else {
    Run $vpy @("-m", "pip", "install", "-q", "--force-reinstall", $wheelPath)
}
Run $vpy @("-c", "import habitat_sim; print('habitat_sim', habitat_sim.__version__, 'bullet:', habitat_sim.built_with_bullet)")

# test scripts from the repo (same layout as the repo so they find data/)
$tools = Join-Path $Dir "tools\windows"
New-Item -ItemType Directory -Force -Path $tools | Out-Null
foreach ($f in @("smoke_test.py", "humanoid_test.py", "hab3_humanoid_episode.py", "habitat-lab-windows.patch")) {
    Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/$Repo/$Branch/tools/windows/$f" -OutFile (Join-Path $tools $f)
}

# ---------------------------------------------------------------- 3. data
Step "3/5 Downloading test scenes and humanoids"
Push-Location $Dir
try {
    Run $vpy @("-m", "habitat_sim.utils.datasets_download", "--uids", "habitat_test_scenes", "habitat_humanoids", "--data-path", "data/", "--no-replace")
} finally { Pop-Location }

# ---------------------------------------------------------------- 4. tests
Step "4/5 Running habitat-sim tests on the GPU"
$env:PYTHONUTF8 = "1"
Push-Location $Dir
try {
    $code = RunLogged "$Results\smoke_log.txt" @("tools\windows\smoke_test.py", "--out", "$Results\smoke")
    $summary["Smoke test"] = if ($code -eq 0) { "PASS" } else { "FAIL (exit $code)" }
    $code = RunLogged "$Results\humanoid_log.txt" @("tools\windows\humanoid_test.py", "--out", "$Results\humanoid")
    $summary["Humanoid test"] = if ($code -eq 0) { "PASS" } else { "FAIL (exit $code)" }
} finally { Pop-Location }

$renderer = Select-String -Path "$Results\smoke_log.txt" -Pattern "^Renderer:" | Select-Object -First 1
$summary["OpenGL renderer"] = if ($renderer) { $renderer.Line } else { "unknown" }

# ---------------------------------------------------------------- 5. habitat-lab (optional)
if ($WithHabitatLab) {
    Step "5/5 Habitat-Lab + Habitat 3.0 humanoid task"
    $lab = Join-Path $Dir "habitat-lab"
    if (-not (Test-Path "$lab\.git")) { Run "git" @("clone", "-q", "https://github.com/facebookresearch/habitat-lab.git", $lab) }
    Push-Location $lab
    try {
        Run "git" @("checkout", "-q", "-f", $HabitatLabSha)
        Run "git" @("apply", "--whitespace=nowarn", (Join-Path $tools "habitat-lab-windows.patch"))
        Run $vpy @("-m", "pip", "install", "-q", "-c", (Join-Path $Dir "constraints.txt"), "-e", "habitat-lab", "-e", "habitat-baselines")
        Run $vpy @("-m", "habitat_sim.utils.datasets_download", "--uids", "hab3_bench_assets", "ycb", "--data-path", "data/", "--no-replace")
        $code = RunLogged "$Results\hab3_log.txt" @((Join-Path $tools "hab3_humanoid_episode.py"), "--out", "$Results\hab3")
        $summary["Habitat 3.0 episode"] = if ($code -eq 0) { "PASS" } else { "FAIL (exit $code)" }
        $null = RunLogged "$Results\hab3_bench_log.txt" @("scripts\hab3_bench\hab3_benchmark.py",
            "--cfg", "benchmark/rearrange/hab3_bench/humanoid_oracle.yaml", "--n-steps", "300", "--n-procs", "1", "--out-name", "human_oracle_win",
            "habitat.task.task_spec=tidy_house_2obj", "habitat.task.pddl_domain_def=fp",
            "habitat.dataset.data_path=data/hab3_bench_assets/episode_datasets/small_small.json.gz",
            "habitat.dataset.scenes_dir=data/hab3_bench_assets")
        $fps = Select-String -Path "$Results\hab3_bench_log.txt" -Pattern "average FPS" | Select-Object -Last 1
        $summary["Habitat 3.0 benchmark"] = if ($fps) { $fps.Line.Trim() } else { "no result" }
    } catch {
        $summary["Habitat-Lab"] = "FAIL: $_"
    } finally { Pop-Location }
} else {
    Step "5/5 Habitat-Lab skipped (run again with -WithHabitatLab to include it)"
}

# ---------------------------------------------------------------- report
Step "Summary"
foreach ($k in $summary.Keys) { Write-Host ("{0,-24} {1}" -f $k, $summary[$k]) }
foreach ($r in @("$Results\smoke\report.txt", "$Results\humanoid\report.txt", "$Results\hab3\report.txt")) {
    if (Test-Path $r) { Write-Host ""; Get-Content $r | Out-Host }
}
$summary.GetEnumerator() | ForEach-Object { "$($_.Key): $($_.Value)" } | Set-Content (Join-Path $Results "summary.txt")

Stop-Transcript | Out-Null
$zip = Join-Path $Dir "habitat_gpu_test_results.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path "$Results\*" -DestinationPath $zip
Write-Host ""
Write-Host "Done. Please send this file back: $zip" -ForegroundColor Green
