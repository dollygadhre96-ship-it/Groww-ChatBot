$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ProjectPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogDirectory = Join-Path $env:LOCALAPPDATA "Groww ChatBot\logs"
$StdoutLog = Join-Path $LogDirectory "streamlit.out.log"
$StderrLog = Join-Path $LogDirectory "streamlit.err.log"

if (-not (Test-Path -LiteralPath $ProjectPython)) {
    throw "Project virtual environment is missing: $ProjectPython"
}

# Leave a healthy server alone. The scheduled task runs this check at sign-in
# and once a minute, so an exited server is brought back without user action.
try {
    $response = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8501/" -TimeoutSec 3
    if ($response.StatusCode -eq 200) { exit 0 }
} catch {
    # No healthy app response; check for a stale listener before relaunching.
}

$listeners = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue
foreach ($listener in $listeners) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)" -ErrorAction SilentlyContinue
    if ($process -and $process.CommandLine -match "streamlit.*app/main\.py") {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    } else {
        throw "Port 8501 is occupied by another process; Groww ChatBot was not started."
    }
}

New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
Start-Process -FilePath $ProjectPython `
    -ArgumentList "-m streamlit run app/main.py --server.headless true --server.address 127.0.0.1 --server.port 8501" `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog
