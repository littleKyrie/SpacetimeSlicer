[CmdletBinding()]
param(
    [string]$EnvironmentName = ".venv",
    [ValidateSet("cu128", "cu126", "cu118", "cpu")]
    [string]$TorchBackend = "cu128"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = `
    [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ThirdPartyRoot = Join-Path $RepoRoot "third_party"

# Suppress uv progress bars (they go to stderr and trip PowerShell's error handling)
$env:UV_NO_PROGRESS = "1"

# Network resilience: increase timeout and use Tsinghua mirror (much faster inside China)
$env:UV_HTTP_TIMEOUT = "300"
$env:UV_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"

# Resolve the venv directory — relative paths are relative to the repo root.
if ([System.IO.Path]::IsPathRooted($EnvironmentName)) {
    $VenvPath = $EnvironmentName
}
else {
    $VenvPath = Join-Path $RepoRoot $EnvironmentName
}
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

$RvmRepository = "https://github.com/PeterL1n/RobustVideoMatting.git"
$RvmRevision = "17d1774"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Ensure-GitRepository {
    param(
        [string]$Path,
        [string]$Url,
        [string]$Revision
    )

    $gitMetadata = Join-Path $Path ".git"
    if (-not (Test-Path -LiteralPath $gitMetadata)) {
        if (Test-Path -LiteralPath $Path) {
            $entries = @(Get-ChildItem -LiteralPath $Path -Force)
            if ($entries.Count -gt 0) {
                throw "Cannot clone into non-empty directory without Git metadata: $Path"
            }
            Remove-Item -LiteralPath $Path -Force
        }
        git clone $Url $Path
        if ($LASTEXITCODE -ne 0) { throw "Failed to clone $Url" }
    }

    $currentRevision = (& git -C $Path rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect Git repository: $Path"
    }
    if (-not $currentRevision.StartsWith($Revision, [System.StringComparison]::OrdinalIgnoreCase)) {
        git -C $Path fetch origin --tags
        git -C $Path checkout --detach $Revision
        if ($LASTEXITCODE -ne 0) { throw "Failed to checkout $Revision in $Path" }
    }
}

# ---- Prerequisite checks ----

if ($env:OS -ne "Windows_NT") {
    throw "setup_lite.ps1 currently supports Windows only."
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was not found. Install it with: powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`" and reopen PowerShell."
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git was not found. Install Git for Windows and reopen PowerShell."
}

Set-Location $RepoRoot
Ensure-Directory $ThirdPartyRoot

# ---- 1. Create virtual environment ----

Write-Step "Creating or reusing uv virtual environment at '$VenvPath'"

# Ensure Python 3.10 is available for uv
cmd /c "uv python install 3.10 2>nul"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    uv venv --python 3.10 $VenvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment: $VenvPath"
    }
}
else {
    Write-Host "Virtual environment already exists: $VenvPath"
}

# Let uv auto-detect the venv for all subsequent pip commands
$env:VIRTUAL_ENV = $VenvPath

# ---- 2. Install Python packages (RVM-only minimum) ----

Write-Step "Installing Python packages"

uv pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip/setuptools/wheel" }

$torchIndexUrl = "https://download.pytorch.org/whl/$TorchBackend"
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 `
    --index-url $torchIndexUrl
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed" }

uv pip install `
    "numpy>=2.2,<2.3" `
    "opencv-contrib-python>=4.10,<5" `
    "pillow>=10,<13"
if ($LASTEXITCODE -ne 0) { throw "Python dependency install failed" }

# ---- 3. torchvision compatibility shim ----

Write-Step "Creating torchvision compatibility shim"
$torchvisionUtilsPath = & $VenvPython -c "import torchvision.models, os; print(os.path.join(os.path.dirname(torchvision.models.__file__), 'utils.py'))"
if ($LASTEXITCODE -eq 0 -and (-not (Test-Path -LiteralPath $torchvisionUtilsPath))) {
    @"
# Compatibility shim: torchvision>=0.22 removed models.utils.
# The RVM repo (pinned to an old revision) imports from here.
from torch.hub import load_state_dict_from_url
"@ | Out-File -FilePath $torchvisionUtilsPath -Encoding utf8
    Write-Host "Created: $torchvisionUtilsPath"
}
elseif (Test-Path -LiteralPath $torchvisionUtilsPath) {
    Write-Host "Already present: $torchvisionUtilsPath"
}

# ---- 4. Clone RVM repository ----

Write-Step "Preparing RobustVideoMatting (RVM segmentation)"
$rvmPath = Join-Path $ThirdPartyRoot "RobustVideoMatting"
Ensure-GitRepository -Path $rvmPath -Url $RvmRepository -Revision $RvmRevision

# ---- 5. Validate ----

Write-Step "Validating the environment"

& $VenvPython -c `
    "import cv2, numpy, PIL, torch, torchvision; print('Python imports OK'); print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "Python import validation failed" }

& $VenvPython build_spacetime_slicer.py --help
if ($LASTEXITCODE -ne 0) { throw "build_spacetime_slicer.py --help failed" }

& $VenvPython batch_run.py --help
if ($LASTEXITCODE -ne 0) { throw "batch_run.py --help failed" }

Write-Host ""
Write-Host "RVM-only environment setup completed." -ForegroundColor Green
Write-Host "Python: $VenvPython"
Write-Host "To enter the venv, run: $VenvPath\Scripts\activate"
Write-Host ""
Write-Host "Note: this lite setup only supports --method RVM (the default)."
Write-Host "      For Hybrid / SAM2_BBox / RMBG2 / rembg, use the full setup.ps1."
