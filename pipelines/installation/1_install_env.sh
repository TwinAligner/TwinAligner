export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
# Define source and target paths for clarity
SOURCE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include/crt"
TARGET_LINK="$CONDA_PREFIX/include/crt"

# Check if the TARGET_LINK already exists (-e checks for file, directory, or link existence)
if [ -e "$TARGET_LINK" ]; then
    echo "Target path $TARGET_LINK already exists. Skipping symbolic link creation."
else
    # Check if the source path actually exists before attempting to link (optional but safe)
    if [ ! -e "$SOURCE_PATH" ]; then
        echo "Error: Source path $SOURCE_PATH does not exist. Cannot create link."
        exit 1
    fi
    
    # Create the symbolic link
    echo "Creating symbolic link: $SOURCE_PATH -> $TARGET_LINK"
    ln -s "$SOURCE_PATH" "$TARGET_LINK"
    
    # Check if the symbolic link creation command was successful (exit code 0)
    if [ $? -eq 0 ]; then
        echo "Symbolic link successfully created."
    else
        echo "Error: Failed to create symbolic link!"
    fi
fi

pip install evdev==1.9.2
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0+cu121 torchvision==0.19.0+cu121
uv pip install --no-build-isolation -r requirements.txt