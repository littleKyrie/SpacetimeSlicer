[CmdletBinding()]
param(
    [string]$EnvironmentName = ".venv",
    [ValidateSet("cu128", "cu126", "cu118", "cpu")]
    [string]$TorchBackend = "cu128",
    [switch]$SkipModels,
    [switch]$SkipRife
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = `
    [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ThirdPartyRoot = Join-Path $RepoRoot "third_party"
$CheckpointRoot = Join-Path $RepoRoot "checkpoints"

# Suppress uv progress bars (they go to stderr and trip PowerShell's error handling)
$env:UV_NO_PROGRESS = "1"

# Resolve the venv directory — relative paths are relative to the repo root.
if ([System.IO.Path]::IsPathRooted($EnvironmentName)) {
    $VenvPath = $EnvironmentName
}
else {
    $VenvPath = Join-Path $RepoRoot $EnvironmentName
}
$VenvPython = Join-Path $VenvPath "Scripts" "python.exe"

$SamRepository = "https://github.com/facebookresearch/sam2.git"
$SamRevision = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
$RvmRepository = "https://github.com/PeterL1n/RobustVideoMatting.git"
$RvmRevision = "17d1774"

$SamLargeUrl = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
$YoloUrl = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
$RvmWeightUrl = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth"
$RifeUrl = "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-windows.zip"

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

function Download-File {
    param(
        [string]$Url,
        [string]$Destination,
        [long]$MinimumBytes = 1
    )

    if (Test-Path -LiteralPath $Destination) {
        $existing = Get-Item -LiteralPath $Destination
        if ($existing.Length -ge $MinimumBytes) {
            Write-Host "Already present: $Destination"
            return
        }
    }

    Ensure-Directory (Split-Path -Parent $Destination)
    $temporaryPath = "${Destination}.download"
    if (Test-Path -LiteralPath $temporaryPath) {
        Remove-Item -LiteralPath $temporaryPath -Force
    }

    try {
        Write-Host "Downloading $Url"
        Invoke-WebRequest -Uri $Url -OutFile $temporaryPath -UseBasicParsing
        $downloaded = Get-Item -LiteralPath $temporaryPath
        if ($downloaded.Length -lt $MinimumBytes) {
            throw "Downloaded file is unexpectedly small: $temporaryPath"
        }
        Move-Item -LiteralPath $temporaryPath -Destination $Destination -Force
    }
    finally {
        if (Test-Path -LiteralPath $temporaryPath) {
            Remove-Item -LiteralPath $temporaryPath -Force
        }
    }
}

# ---- Prerequisite checks ----

if ($env:OS -ne "Windows_NT") {
    throw "setup.ps1 currently supports Windows only."
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was not found. Install it with: powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`" and reopen PowerShell."
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git was not found. Install Git for Windows and reopen PowerShell."
}

Set-Location $RepoRoot
Ensure-Directory $ThirdPartyRoot
Ensure-Directory $CheckpointRoot

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

# ---- 2. Install Python packages ----

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
    "pillow>=10,<13" `
    "ultralytics>=8.3,<9" `
    "transformers>=4.45,<6" `
    "hydra-core>=1.3,<2" `
    "iopath>=0.1.10,<0.2" `
    "huggingface-hub>=0.26" `
    "safetensors>=0.4" `
    "kornia>=0.7,<1" `
    "scipy>=1.14,<2" `
    "matplotlib>=3.9,<4" `
    "tqdm>=4.66" `
    "pytest>=8,<10"
if ($LASTEXITCODE -ne 0) { throw "Python dependency install failed" }

# ---- 3. Clone and install third-party repositories ----

Write-Step "Preparing pinned third-party source repositories"
$samPath = Join-Path $ThirdPartyRoot "sam2"
$rvmPath = Join-Path $ThirdPartyRoot "RobustVideoMatting"
Ensure-GitRepository -Path $samPath -Url $SamRepository -Revision $SamRevision
Ensure-GitRepository -Path $rvmPath -Url $RvmRepository -Revision $RvmRevision
uv pip install -e $samPath
if ($LASTEXITCODE -ne 0) { throw "SAM2 editable install failed" }

# ---- 4. Download model weights ----

if (-not $SkipModels) {
    Write-Step "Downloading required model weights"
    $samCheckpoint = Join-Path $CheckpointRoot "sam2\sam2.1_hiera_large.pt"
    $yoloCheckpoint = Join-Path $CheckpointRoot "yolo\yolov8n.pt"
    Download-File -Url $SamLargeUrl -Destination $samCheckpoint -MinimumBytes 800000000
    Download-File -Url $YoloUrl -Destination $yoloCheckpoint -MinimumBytes 5000000

    $torchHubOutput = & $VenvPython -c "import torch; print(torch.hub.get_dir())"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to determine the Torch Hub cache directory."
    }
    $torchHubDirectory = (
        $torchHubOutput | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        Select-Object -Last 1
    ).Trim()
    $rvmCheckpoint = Join-Path $torchHubDirectory "checkpoints\rvm_resnet50.pth"
    Download-File -Url $RvmWeightUrl -Destination $rvmCheckpoint -MinimumBytes 100000000
}
else {
    Write-Host "Model downloads skipped by -SkipModels."
}

# ---- 5. Download RIFE ----

if (-not $SkipRife) {
    Write-Step "Preparing rife-ncnn-vulkan"
    $rifeTarget = Join-Path $ThirdPartyRoot "rife-ncnn-vulkan"
    $rifeExecutable = Join-Path $rifeTarget "rife-ncnn-vulkan.exe"
    if (-not (Test-Path -LiteralPath $rifeExecutable)) {
        $temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("SpacetimeSlicer-" + [guid]::NewGuid())
        $archivePath = Join-Path $temporaryRoot "rife.zip"
        $extractPath = Join-Path $temporaryRoot "expanded"
        try {
            Ensure-Directory $temporaryRoot
            Download-File -Url $RifeUrl -Destination $archivePath -MinimumBytes 1000000
            Expand-Archive -LiteralPath $archivePath -DestinationPath $extractPath -Force
            $downloadedExecutable = Get-ChildItem -LiteralPath $extractPath -Recurse -File `
                -Filter "rife-ncnn-vulkan.exe" | Select-Object -First 1
            if ($null -eq $downloadedExecutable) {
                throw "The RIFE archive did not contain rife-ncnn-vulkan.exe."
            }
            Ensure-Directory $rifeTarget
            Copy-Item -Path (Join-Path $downloadedExecutable.Directory.FullName "*") `
                -Destination $rifeTarget -Recurse -Force
        }
        finally {
            if (Test-Path -LiteralPath $temporaryRoot) {
                Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
            }
        }
    }
    else {
        Write-Host "Already present: $rifeExecutable"
    }
}
else {
    Write-Host "RIFE setup skipped by -SkipRife."
}

# ---- 6. Validate ----

Write-Step "Validating the environment"

& $VenvPython -c `
    "import cv2, numpy, PIL, torch, torchvision, ultralytics, transformers, sam2; print('Python imports OK'); print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "Python import validation failed" }

& $VenvPython build_spacetime_slicer.py --help
if ($LASTEXITCODE -ne 0) { throw "build_spacetime_slicer.py --help failed" }

& $VenvPython batch_run.py --help
if ($LASTEXITCODE -ne 0) { throw "batch_run.py --help failed" }

Write-Host ""
Write-Host "Environment setup completed." -ForegroundColor Green
$activateScript = Join-Path $VenvPath "Scripts" "activate"
Write-Host "Activate it with: $activateScript"
if ($SkipModels) {
    Write-Host "Required model weights were not downloaded because -SkipModels was used."
}
if ($SkipRife) {
    Write-Host "RIFE was not installed because -SkipRife was used."
}
