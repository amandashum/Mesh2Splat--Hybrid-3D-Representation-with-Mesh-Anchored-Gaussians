param(
    [Parameter(Mandatory = $true)]
    [string]$ColmapBat,

    [Parameter(Mandatory = $true)]
    [string]$ImageDir,

    [Parameter(Mandatory = $true)]
    [string]$Workspace,

    [string]$CameraModel = "SIMPLE_RADIAL",
    [int]$UseGpu = 1
)

$ErrorActionPreference = "Stop"

$colmapBatResolved = (Resolve-Path $ColmapBat).Path
$imageDirResolved = (Resolve-Path $ImageDir).Path

if (-not (Test-Path $Workspace)) {
    New-Item -ItemType Directory -Path $Workspace | Out-Null
}

$workspaceResolved = (Resolve-Path $Workspace).Path
$databasePath = Join-Path $workspaceResolved "database.db"
$sparseDir = Join-Path $workspaceResolved "sparse"
$denseDir = Join-Path $workspaceResolved "dense"
$sparseModelDir = Join-Path $sparseDir "0"
$sparseTxtDir = Join-Path $workspaceResolved "sparse_txt"

# Build a standard COLMAP workspace that this repo can consume directly. The
# output layout mirrors what the Python loader expects: a sparse text model and
# the corresponding image directory.
foreach ($dir in @($sparseDir, $denseDir, $sparseTxtDir)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
    }
}

Write-Host "Running feature extraction..."
& $colmapBatResolved feature_extractor `
    --database_path $databasePath `
    --image_path $imageDirResolved `
    --ImageReader.camera_model $CameraModel `
    --FeatureExtraction.use_gpu $UseGpu

Write-Host "Running exhaustive matching..."
& $colmapBatResolved exhaustive_matcher `
    --database_path $databasePath `
    --FeatureMatching.use_gpu $UseGpu

Write-Host "Running sparse mapping..."
& $colmapBatResolved mapper `
    --database_path $databasePath `
    --image_path $imageDirResolved `
    --output_path $sparseDir

if (-not (Test-Path $sparseModelDir)) {
    throw "COLMAP mapper did not produce $sparseModelDir"
}

Write-Host "Running image undistortion..."
& $colmapBatResolved image_undistorter `
    --image_path $imageDirResolved `
    --input_path $sparseModelDir `
    --output_path $denseDir `
    --output_type COLMAP

Write-Host "Exporting sparse model to TXT..."
& $colmapBatResolved model_converter `
    --input_path $sparseModelDir `
    --output_path $sparseTxtDir `
    --output_type TXT

Write-Host ""
Write-Host "COLMAP finished."
Write-Host "Text model: $sparseTxtDir"
Write-Host "Images:     $imageDirResolved"
