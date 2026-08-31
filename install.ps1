[CmdletBinding()]
param(
    [string]$Release = "latest",
    [string]$InstallRoot = "$env:ProgramFiles\CTOX",
    [string]$StateRoot = "$env:ProgramData\CTOX",
    [switch]$NoStart
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Write-CtoxEvent {
    param(
        [Parameter(Mandatory = $true)][string]$Phase,
        [Parameter(Mandatory = $true)][string]$Status,
        [Parameter(Mandatory = $true)][string]$Message,
        [int]$Percent = 0
    )
    [ordered]@{
        schema = "ctox.install-event.v1"
        phase = $Phase
        status = $Status
        percent = $Percent
        message = $Message
        timestamp = [DateTimeOffset]::UtcNow.ToString("o")
    } | ConvertTo-Json -Compress | Write-Output
}

function Assert-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "CTOX installation requires an administrator PowerShell session"
    }
}

function Remove-PathIfPresent {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

$tempRoot = $null
try {
    Write-CtoxEvent -Phase "preflight" -Status "started" -Message "Checking Windows target" -Percent 2
    Assert-Administrator
    if ([Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne [Runtime.InteropServices.Architecture]::X64) {
        throw "CTOX Windows releases currently support x64 only"
    }
    $releasePath = if ($Release -eq "latest") {
        "latest/download"
    } else {
        "download/$([Uri]::EscapeDataString($Release))"
    }
    $manifestUrl = "https://github.com/metric-space-ai/ctox/releases/$releasePath/ctox-install-manifest-v1.json"
    $tempRoot = Join-Path ([IO.Path]::GetTempPath()) ("ctox-install-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $tempRoot | Out-Null
    $manifestPath = Join-Path $tempRoot "ctox-install-manifest-v1.json"
    Invoke-WebRequest -UseBasicParsing -Uri $manifestUrl -OutFile $manifestPath
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.schema -ne "ctox.install-manifest.v1" -or $manifest.version -ne 1) {
        throw "Unsupported CTOX install manifest"
    }
    if ($manifest.repository -ne "metric-space-ai/ctox") {
        throw "Refusing CTOX manifest from an unexpected repository"
    }
    $artifact = @($manifest.artifacts | Where-Object { $_.platform -eq "windows" -and $_.arch -eq "x64" })
    if ($artifact.Count -ne 1) {
        throw "CTOX manifest has no unique windows/x64 artifact"
    }
    $artifact = $artifact[0]
    $artifactUri = [Uri]$artifact.url
    if ($artifactUri.Scheme -ne "https" -or $artifactUri.Host -ne "github.com") {
        throw "Refusing a non-GitHub or non-HTTPS CTOX artifact URL"
    }
    Write-CtoxEvent -Phase "preflight" -Status "completed" -Message "Administrator, internet and release manifest verified" -Percent 10

    $archivePath = Join-Path $tempRoot $artifact.filename
    Write-CtoxEvent -Phase "download" -Status "started" -Message "Downloading CTOX $($manifest.release)" -Percent 15
    Invoke-WebRequest -UseBasicParsing -Uri $artifact.url -OutFile $archivePath
    Write-CtoxEvent -Phase "download" -Status "completed" -Message "CTOX archive downloaded" -Percent 40
    $actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne ([string]$artifact.sha256).ToLowerInvariant()) {
        throw "CTOX archive SHA-256 mismatch"
    }
    Write-CtoxEvent -Phase "verify" -Status "completed" -Message "CTOX archive SHA-256 verified" -Percent 48

    New-Item -ItemType Directory -Force -Path $InstallRoot, $StateRoot, (Join-Path $InstallRoot "releases") | Out-Null
    $releaseName = ([string]$manifest.release) -replace '[^A-Za-z0-9._-]', '_'
    $releaseRoot = Join-Path $InstallRoot "releases\$releaseName"
    $stagingRoot = "$releaseRoot.staging"
    Remove-PathIfPresent $stagingRoot
    New-Item -ItemType Directory -Path $stagingRoot | Out-Null
    Expand-Archive -LiteralPath $archivePath -DestinationPath $stagingRoot -Force
    if (-not (Test-Path -LiteralPath (Join-Path $stagingRoot "bin\ctox.exe") -PathType Leaf)) {
        throw "CTOX archive is missing bin\ctox.exe"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $stagingRoot "contracts\binary_bundle_manifest.txt") -PathType Leaf)) {
        throw "CTOX archive is missing its binary bundle contract"
    }

    $currentRoot = Join-Path $InstallRoot "current"
    $currentBinary = Join-Path $currentRoot "bin\ctox.exe"
    if (Test-Path -LiteralPath $currentBinary) {
        & $currentBinary stop --force 2>$null | Out-Null
    }
    Remove-PathIfPresent $releaseRoot
    Move-Item -LiteralPath $stagingRoot -Destination $releaseRoot
    $nextLink = Join-Path $InstallRoot "current.next"
    $oldLink = Join-Path $InstallRoot "current.previous-link"
    Remove-PathIfPresent $nextLink
    Remove-PathIfPresent $oldLink
    New-Item -ItemType Junction -Path $nextLink -Target $releaseRoot | Out-Null
    if (Test-Path -LiteralPath $currentRoot) {
        Move-Item -LiteralPath $currentRoot -Destination $oldLink
    }
    Move-Item -LiteralPath $nextLink -Destination $currentRoot
    Remove-PathIfPresent $oldLink
    Write-CtoxEvent -Phase "install" -Status "completed" -Message "CTOX release activated" -Percent 70

    $manifestState = [ordered]@{
        schema_version = 1
        install_root = $InstallRoot
        state_root = $StateRoot
        current_release = [string]$manifest.release
        previous_release = $null
        adopted_from = $null
        release_channel = [ordered]@{
            kind = "github"
            repo = "metric-space-ai/ctox"
            api_base = "https://api.github.com"
            token_env = $null
        }
        updated_at = [DateTimeOffset]::UtcNow.ToString("o")
    }
    $manifestState | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $InstallRoot "install_manifest.json") -Encoding UTF8

    $ctox = Join-Path $currentRoot "bin\ctox.exe"
    Write-CtoxEvent -Phase "service" -Status "started" -Message "Installing CTOX Windows service" -Percent 75
    & $ctox service install --root $currentRoot | Out-Null
    New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\CTOX" -Name "Environment" -PropertyType MultiString -Force -Value @(
        "CTOX_ROOT=$currentRoot",
        "CTOX_INSTALL_ROOT=$InstallRoot",
        "CTOX_STATE_ROOT=$StateRoot",
        "CTOX_CACHE_ROOT=$(Join-Path $StateRoot 'cache')"
    ) | Out-Null
    if (-not $NoStart) {
        & $ctox start | Out-Null
        $status = & $ctox status | ConvertFrom-Json
        if (-not $status.running) {
            throw "CTOX service did not report healthy after installation"
        }
    }
    Write-CtoxEvent -Phase "service" -Status "completed" -Message "CTOX Windows service installed" -Percent 95
    Write-CtoxEvent -Phase "complete" -Status "completed" -Message "CTOX backend is installed" -Percent 100
} catch {
    Write-CtoxEvent -Phase "failed" -Status "failed" -Message $_.Exception.Message -Percent 0
    exit 1
} finally {
    if ($tempRoot) {
        Remove-PathIfPresent $tempRoot
    }
}
