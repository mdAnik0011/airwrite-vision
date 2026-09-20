# Launch from any directory. A normal project venv takes precedence.
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $airwritePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $airwritePython)) {
        $airwritePython = Join-Path $PSScriptRoot '..\.tools\python\python.exe'
    }
    if (-not (Test-Path -LiteralPath $airwritePython)) {
        $airwritePython = 'python'
    }
    & $airwritePython -m airwrite.app @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
