#!/bin/bash

# remove logs
rm *log

# remove downloads
rm downloads/*

# kill tmux session
tmux kill-session -t openvpn_session

# kill openvpn process
sudo kill $(pgrep openvpn)