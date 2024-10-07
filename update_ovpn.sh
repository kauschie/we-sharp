#!/bin/bash

# Path to the directory where your .ovpn files are located
OVPN_DIR="/home/mkausch/dev/we-sharp/pia"
CREDENTIALS_PATH="/etc/openvpn/pia-credentials.txt"

# Loop through each .ovpn file
for file in "$OVPN_DIR"/*.ovpn; do
  # Check if the auth-user-pass line already exists
  if grep -q "auth-user-pass" "$file"; then
    # Replace the existing auth-user-pass line with the one pointing to the credentials file
    sed -i "s|auth-user-pass.*|auth-user-pass $CREDENTIALS_PATH|" "$file"
    echo "Updated auth-user-pass in $file to point to $CREDENTIALS_PATH."
  else
    # If auth-user-pass does not exist, add it at the end of the file
    echo "auth-user-pass $CREDENTIALS_PATH" >> "$file"
    echo "Added auth-user-pass line to $file."
  fi
done
