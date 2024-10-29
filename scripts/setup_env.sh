#!/bin/bash

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if requirements.txt exists)
if [[ -f "requirements.txt" ]]; then
  pip install -r requirements.txt
fi

# Define the project root in .bashrc
if ! grep -q "MY_PROJ_ROOT" ~/.bashrc; then
  echo "export MY_PROJ_ROOT=$(pwd)" >> ~/.bashrc
  echo "Project root directory set to $(pwd)"
fi

# Add the toggle function and freeze alias to .bashrc if not already present
if ! grep -q "MYENV_TOGGLE_SETUP" ~/.bashrc; then
  echo "# MYENV_TOGGLE_SETUP - Virtual environment setup" >> ~/.bashrc

  # Add the toggle function for activation/deactivation
  echo "toggle_myenv() {" >> ~/.bashrc
  echo "  if [[ -z \"\$VIRTUAL_ENV\" ]]; then" >> ~/.bashrc
  echo "    source \$MY_PROJ_ROOT/.venv/bin/activate" >> ~/.bashrc
  echo "    echo 'Virtual environment activated.'" >> ~/.bashrc
  echo "  else" >> ~/.bashrc
  echo "    deactivate" >> ~/.bashrc
  echo "    echo 'Virtual environment deactivated.'" >> ~/.bashrc
  echo "  fi" >> ~/.bashrc
  echo "}" >> ~/.bashrc
  echo "alias toggle_myenv=toggle_myenv" >> ~/.bashrc

  # Add the freeze alias, using the MY_PROJ_ROOT variable
  echo "alias freeze='pip freeze > \$MY_PROJ_ROOT/requirements.txt && echo \"requirements.txt updated\"'" >> ~/.bashrc

  echo "# MYENV_TOGGLE_SETUP end" >> ~/.bashrc

  # Source the .bashrc to make everything available in the current session
  source ~/.bashrc

  echo "Setup complete. Use 'toggle_myenv' to activate/deactivate the environment and 'freeze' to update requirements.txt."
else
  echo "The virtual environment toggle and freeze alias are already set up."
fi