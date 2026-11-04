$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnsureScript = Join-Path $PSScriptRoot "ensure_app.ps1"
$TaskName = "Groww ChatBot Streamlit Auto Start"
$User = "$env:USERDOMAIN\$env:USERNAME"
$WindowsPowerShell = Join-Path $env:WINDIR "System32\WindowsPowerShell\v1.0\powershell.exe"

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot ".venv\Scripts\python.exe"))) {
    throw "Project virtual environment is missing. Install requirements.txt before configuring automatic startup."
}

$LogonTrigger = New-ScheduledTaskTrigger -AtLogOn -User $User
$RecoveryTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 1) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$Action = New-ScheduledTaskAction `
    -Execute $WindowsPowerShell `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$EnsureScript`"" `
    -WorkingDirectory $ProjectRoot
$Principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet `
    -Hidden `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 30) `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger @($LogonTrigger, $RecoveryTrigger) `
    -Principal $Principal `
    -Settings $Settings `
    -Description "Starts Groww ChatBot at sign-in and checks once per minute that the local Streamlit page is available." `
    -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName
Write-Host "Automatic startup and recovery enabled for Groww ChatBot."
