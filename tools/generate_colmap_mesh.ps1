param(
    [Parameter(Mandatory = $true)]
    [string]$ColmapBat,

    [Parameter(Mandatory = $true)]
    [string]$ImageDir,

    [Parameter(Mandatory = $true)]
    [string]$ModelDir,

    [Parameter(Mandatory = $true)]
    [string]$Workspace,

    [ValidateSet("poisson", "delaunay")]
    [string]$Mesher = "poisson",

    [int]$MaxImageSize = 1600,

    [switch]$UseCpu
)

$ErrorActionPreference = "Stop"

function Invoke-ColmapStep {
    param(
        [string]$Label,
        [string[]]$Arguments
    )

    Write-Host "Running $Label..."
    & $ColmapBat @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "COLMAP step failed: $Label"
    }
}

$workspacePath = [System.IO.Path]::GetFullPath($Workspace)
$imagePath = (Resolve-Path -LiteralPath $ImageDir).Path
$modelPath = (Resolve-Path -LiteralPath $ModelDir).Path
$densePath = Join-Path $workspacePath "dense"
$fusedPath = Join-Path $densePath "fused.ply"
$meshPlyPath = if ($Mesher -eq "poisson") {
    Join-Path $densePath "meshed-poisson.ply"
} else {
    Join-Path $densePath "meshed-delaunay.ply"
}
$meshObjPath = Join-Path $densePath "mesh_prior.obj"

New-Item -ItemType Directory -Path $densePath -Force | Out-Null
$patchMatchGpuIndex = if ($UseCpu.IsPresent) { "-1" } else { "0" }

# Dense reconstruction begins from an existing sparse COLMAP model. This keeps
# the mesh-prior generation step separate from training and makes it reusable.
Invoke-ColmapStep -Label "image undistortion" -Arguments @(
    "image_undistorter",
    "--image_path", $imagePath,
    "--input_path", $modelPath,
    "--output_path", $densePath,
    "--output_type", "COLMAP",
    "--max_image_size", "$MaxImageSize"
)

Invoke-ColmapStep -Label "patch match stereo" -Arguments @(
    "patch_match_stereo",
    "--workspace_path", $densePath,
    "--workspace_format", "COLMAP",
    "--PatchMatchStereo.geom_consistency", "true",
    "--PatchMatchStereo.gpu_index", $patchMatchGpuIndex
)

Invoke-ColmapStep -Label "stereo fusion" -Arguments @(
    "stereo_fusion",
    "--workspace_path", $densePath,
    "--workspace_format", "COLMAP",
    "--input_type", "geometric",
    "--output_path", $fusedPath
)

if ($Mesher -eq "poisson") {
    Invoke-ColmapStep -Label "poisson mesher" -Arguments @(
        "poisson_mesher",
        "--input_path", $fusedPath,
        "--output_path", $meshPlyPath
    )
} else {
    Invoke-ColmapStep -Label "delaunay mesher" -Arguments @(
        "delaunay_mesher",
        "--input_path", $densePath,
        "--output_path", $meshPlyPath
    )
}

Write-Host "Converting mesh PLY to OBJ..."
python .\tools\ply_to_obj.py --input $meshPlyPath --output $meshObjPath
if ($LASTEXITCODE -ne 0) {
    throw "PLY to OBJ conversion failed."
}

Write-Host "Mesh prior OBJ written to $meshObjPath"
