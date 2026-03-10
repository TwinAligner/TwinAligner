#!/bin/bash
uv pip install gdown

# 1. Create the main directory (mkdir -p is already idempotent/safe)
mkdir -p checkpoints

# ----------------------------------------------------
# 2. Download depth_anything_v2_vitl.pth
# Check if the target file exists.
TARGET_FILE_1="checkpoints/depth_anything_v2_vitl.pth"
if [ ! -f "$TARGET_FILE_1" ]; then
    echo "File $TARGET_FILE_1 not found. Starting download..."
    aria2c -s16 -x16 -k1M https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth -o "$TARGET_FILE_1"
else
    echo "File $TARGET_FILE_1 already exists. Skipping download."
fi

# ----------------------------------------------------
# 3. Download depth_pro.pt
# Check if the target file exists.
TARGET_FILE_2="checkpoints/depth_pro.pt"
if [ ! -f "$TARGET_FILE_2" ]; then
    echo "File $TARGET_FILE_2 not found. Starting download..."
    aria2c -s16 -x16 -k1M https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -o "$TARGET_FILE_2"
else
    echo "File $TARGET_FILE_2 already exists. Skipping download."
fi

# ----------------------------------------------------
# 4. gdown download Google Drive folder contents (FoundationPose)
# We check if the expected output directory, 'checkpoints/FoundationPose', already exists.
TARGET_DIR_GDRIVE="checkpoints/FoundationPose" 
if [ ! -d "$TARGET_DIR_GDRIVE" ]; then
    echo "Expected directory $TARGET_DIR_GDRIVE not found. Starting gdown..."
    # Note: The original command used -O checkpoints/. We assume this gdown operation 
    # results in the creation of the FoundationPose folder within checkpoints/.
    gdown --fuzzy --folder https://drive.google.com/drive/folders/1GdzS948t0e7i3mdL5mMCHmDuO_VerYID?usp=sharing -O checkpoints/
else
    echo "Google Drive content (FoundationPose) seems to be present in $TARGET_DIR_GDRIVE. Skipping gdown."
fi

# ----------------------------------------------------
# 5. Create the sam3 subdirectory (always safe)
mkdir -p checkpoints/sam3

# 6. Install modelscope
# Package installation is generally idempotent and usually doesn't need to be skipped.
uv pip install modelscope

# ----------------------------------------------------
# 7. ModelScope model download
# Check if the model is already present. Models downloaded via ModelScope typically include a config.json file.
TARGET_MODEL_KEY="checkpoints/sam3/config.json"
if [ ! -f "$TARGET_MODEL_KEY" ]; then
    echo "Model facebook/sam3 not found (checked for $TARGET_MODEL_KEY). Starting download..."
    modelscope download --model facebook/sam3 --local_dir checkpoints/sam3
else
    echo "Model facebook/sam3 seems to be present. Skipping download."
fi

# ----------------------------------------------------
# 8. Download coco_lvis_h18_itermask.pth
# Check if the target file exists.
TARGET_FILE_3="checkpoints/coco_lvis_h18_itermask.pth"
if [ ! -f "$TARGET_FILE_3" ]; then
    echo "File $TARGET_FILE_3 not found. Starting download..."
    aria2c -s16 -x16 -k1M https://github.com/hkchengrex/Cutie/releases/download/v1.0/coco_lvis_h18_itermask.pth -o "$TARGET_FILE_3"
else
    echo "File $TARGET_FILE_3 already exists. Skipping download."
fi

# ----------------------------------------------------
# 9. Download cutie-base-mega.pth
# Check if the target file exists.
TARGET_FILE_4="checkpoints/cutie-base-mega.pth"
if [ ! -f "$TARGET_FILE_4" ]; then
    echo "File $TARGET_FILE_4 not found. Starting download..."
    aria2c -s16 -x16 -k1M https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth -o "$TARGET_FILE_4"
else
    echo "File $TARGET_FILE_4 already exists. Skipping download."
fi

# ----------------------------------------------------
# 10. Download Anygrasp checkpoint
# Check if the target file exists.
echo '>>> Downloading Anygrasp Checkpoint ...'
TARGET_FILE_5="checkpoints/checkpoint_detection.tar"
if [ ! -f "$TARGET_FILE_5" ]; then
    echo "File $TARGET_FILE_5 not found. Starting download..."
    gdown 1jNvqOOf_fR3SWkXuz8TAzcHH9x8gE8Et -O checkpoints/checkpoint_detection.tar
else
    echo "File $TARGET_FILE_5 already exists. Skipping download."
fi