$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
function Find-Python {
    foreach ($name in @('py', 'python')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) {
            $detected = & $cmd.Source -c "import sys; print(sys.executable if sys.version_info >= (3,10) else '')" 2>$null
            if ($LASTEXITCODE -eq 0 -and $detected -and (Test-Path -LiteralPath "$detected")) { return "$detected" }
        }
    }
    return $null
}
try {
    $pythonPath = Find-Python
    if (-not $pythonPath) {
        $answer = Read-Host 'Python 3.10+ is required. Install Python 3.12 with winget? [y/N]'
        if ($answer -ne 'y') { throw 'Install Python from python.org, then launch Boute again.' }
        # Installation output goes to the host, never into executable-path detection.
        & winget install --id Python.Python.3.12 --exact --scope user --accept-package-agreements --accept-source-agreements | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'Python installation failed. Install Python from python.org.' }
        $env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
        $pythonPath = Find-Python
        if (-not $pythonPath) { throw 'Python was installed. Close this window and launch Boute again.' }
    }
    & $pythonPath bootstrap.py
    if ($LASTEXITCODE -ne 0) { throw 'Boute could not start. See the error above.' }
} catch {
    Write-Host $_ -ForegroundColor Red
    Read-Host 'Press Enter to close'
    exit 1
}
