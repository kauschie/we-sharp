# Create virtual environment in .env folder
python -m venv .env

# Activate virtual environment
.\.env\Scripts\Activate

# Install dependencies (if requirements.txt exists)
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
}

# Path to the PowerShell profile
$profilePath = [System.Environment]::GetFolderPath("MyDocuments") + "\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"

# Check if the MY_PROJ_ROOT variable is already set in the profile
if (-not (Select-String -Path $profilePath -Pattern "MY_PROJ_ROOT")) {
    # Set the project root directory in the PowerShell profile
    Add-Content $profilePath "`n# MY_PROJ_ROOT - Project root directory"
    Add-Content $profilePath "`$env:MY_PROJ_ROOT = '$(Get-Location)'"
    Write-Host "Project root directory set to $(Get-Location)"
}

# Check if the toggle function and freeze alias have already been added
if (-not (Select-String -Path $profilePath -Pattern "MYENV_TOGGLE_SETUP")) {
    # Add a comment to mark the beginning of the setup
    Add-Content $profilePath "`n# MYENV_TOGGLE_SETUP - Virtual environment setup"

    # Add the toggle function for activation/deactivation
    Add-Content $profilePath "`nfunction Toggle-Venv {"
    Add-Content $profilePath "`n  if (-not `$env:VIRTUAL_ENV) {"
    Add-Content $profilePath "`n    . `$env:MY_PROJ_ROOT\.env\Scripts\Activate"
    Add-Content $profilePath "`n    Write-Host 'Virtual environment activated.'"
    Add-Content $profilePath "`n  } else {"
    Add-Content $profilePath "`n    deactivate"
    Add-Content $profilePath "`n    Write-Host 'Virtual environment deactivated.'"
    Add-Content $profilePath "`n  }"
    Add-Content $profilePath "`n}"
    Add-Content $profilePath "`nSet-Alias toggle_myenv Toggle-Venv"

    # Add the freeze alias to update requirements.txt
    Add-Content $profilePath "`nSet-Alias freeze {pip freeze > `$env:MY_PROJ_ROOT\requirements.txt; Write-Host 'requirements.txt updated'}"

    # Add a comment to mark the end of the setup
    Add-Content $profilePath "`n# MYENV_TOGGLE_SETUP end"

    Write-Host "Virtual environment setup complete. Use 'toggle_myenv' to activate/deactivate and 'freeze' to update requirements.txt."
} else {
    Write-Host "The virtual environment toggle and freeze alias are already set up in your PowerShell profile."
}
