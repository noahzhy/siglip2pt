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
