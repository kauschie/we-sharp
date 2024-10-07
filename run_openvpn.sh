#!/bin/bash

# Ensure we have a config file as an argument
if [ -z "$1" ]; then
    echo "Error: No config file provided."
    exit 1
fi

# Start a new tmux session and run OpenVPN inside it
# The session will be named "openvpn_session"
tmux new-session -d -s openvpn_session "sudo /usr/sbin/openvpn --config $1"
