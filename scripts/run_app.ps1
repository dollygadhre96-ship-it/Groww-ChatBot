$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ProjectPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $ProjectPython)) {
    throw "Project virtual environment is missing. Create it and install requirements.txt before starting the app."
}

Set-Location -LiteralPath $ProjectRoot
& $ProjectPython -m streamlit run app/main.py @args
exit $LASTEXITCODE
