#!/bin/zsh

# Create virtual environment in .env folder
python3 -m venv .env

# Activate virtual environment
source .env/bin/activate

# Install dependencies (if requirements.txt exists)
if [[ -f "requirements.txt" ]]; then
  pip install -r requirements.txt
fi

# Define the project root in .zshrc
if ! grep -q "MY_PROJ_ROOT" ~/.zshrc; then
  echo "export MY_PROJ_ROOT=$(pwd)" >> ~/.zshrc
  echo "Project root directory set to $(pwd)"
fi

# Add the toggle function and freeze alias to .zshrc if not already present
if ! grep -q "MYENV_TOGGLE_SETUP" ~/.zshrc; then
  echo "# MYENV_TOGGLE_SETUP - Virtual environment setup" >> ~/.zshrc

  # Add the toggle function for activation/deactivation
  echo "toggle_myenv() {" >> ~/.zshrc
  echo "  if [[ -z \"\$VIRTUAL_ENV\" ]]; then" >> ~/.zshrc
  echo "    source \$MY_PROJ_ROOT/.env/bin/activate" >> ~/.zshrc
  echo "    echo 'Virtual environment activated.'" >> ~/.zshrc
  echo "  else" >> ~/.zshrc
  echo "    deactivate" >> ~/.zshrc
  echo "    echo 'Virtual environment deactivated.'" >> ~/.zshrc
  echo "  fi" >> ~/.zshrc
  echo "}" >> ~/.zshrc
  echo "alias toggle_myenv=toggle_myenv" >> ~/.zshrc

  # Add the freeze alias, using the MY_PROJ_ROOT variable
  echo "alias freeze='pip freeze > \$MY_PROJ_ROOT/requirements.txt && echo \"requirements.txt updated\"'" >> ~/.zshrc

  echo "# MYENV_TOGGLE_SETUP end" >> ~/.zshrc

  # Source the .zshrc to make everything available in the current session
  source ~/.zshrc

  echo "Setup complete. Use 'toggle_myenv' to activate/deactivate the environment and 'freeze' to update requirements.txt."
else
  echo "The virtual environment toggle and freeze alias are already set up."
fi
