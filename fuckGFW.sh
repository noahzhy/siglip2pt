#!/bin/bash

set -e

# backup current sources.list
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak

# replace with Aliyun mirrors
sudo tee /etc/apt/sources.list > /dev/null <<EOL
deb https://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse
EOL

# update apt
sudo apt update
sudo apt upgrade -y

echo "\33[32m[DONE]\33[0m Apt sources have been replaced with Aliyun mirrors successfully."


# check os type and set profile file
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     profileFile=~/.bashrc;;
    Darwin*)    profileFile=~/.zshrc;;
    *)          profileFile=~/.bashrc;;
esac

# add environment variables if not already present
if ! grep -q "HF_ENDPOINT" "$profileFile"; then
    echo 'export HF_ENDPOINT=https://hf-mirror.com' >> "$profileFile"
    echo 'export HF_HUB_ENABLE_HF_TRANSFER=1' >> "$profileFile"
    echo "Environment variables added to $profileFile."
else
    echo "Environment variables already present in $profileFile."
fi

# source the profile file to apply changes
. "$profileFile"

# check does it install pip install hf_transfer already
if pip show hf_transfer > /dev/null 2>&1; then
    echo "hf_transfer is already installed."
else
    echo "hf_transfer is not installed. Installing..."
    pip install hf_transfer
fi

# done
echo "\33[32m[DONE]\33[0m Let's enjoy huggingface without GFW!"
