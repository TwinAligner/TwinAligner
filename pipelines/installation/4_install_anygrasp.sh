conda install openssl openblas-devel -c anaconda -y
conda install cuda-nvtx -c nvidia -y
uv pip install git+https://github.com/hwfan/graspnetAPI.git
# Install pointnet2
cd simulation/anygrasp_sdk/pointnet2 && uv pip install . --no-build-isolation --verbose && cd ../../..
export BLAS_INCLUDE_DIRS=${CONDA_PREFIX}/include
export BLAS=openblas
uv pip install git+https://github.com/hwfan/MinkowskiEngineCuda13.git@cuda13-installation \
--verbose \
--no-build-isolation

# Copy license
cp -r checkpoints/anygrasp_license simulation/anygrasp_sdk/grasp_detection/license

# Define variables for clearer paths
SOURCE_DIR="simulation/anygrasp_sdk/grasp_detection"
LICENSE_DIR="simulation/anygrasp_sdk/license_registration"

# --- Create Soft Link for the first file: gsnet.so ---

# Define the target path (the desired link name)
TARGET1="${SOURCE_DIR}/gsnet.so"
# Define the source path (the actual file to link to)
SOURCE1="${SOURCE_DIR}/gsnet_versions/gsnet.cpython-310-x86_64-linux-gnu.so"

# Check if the target file (gsnet.so) does NOT exist (-f checks for a regular file)
# The user requested skipping the operation if the file already exists.
if [ ! -f "$TARGET1" ]; then
    echo "Copying: $SOURCE1 --> $TARGET1"
    cp "$SOURCE1" "$TARGET1"
else
    # If the file exists, notify the user and skip the linking step.
    echo "$TARGET1 already exists. Skipping."
fi

# --- Create Soft Link for the second file: lib_cxx.so ---

# Define the target path (the desired link name)
TARGET2="${SOURCE_DIR}/lib_cxx.so"
# Define the source path (the actual file to link to)
SOURCE2="${LICENSE_DIR}/lib_cxx_versions/lib_cxx.cpython-310-x86_64-linux-gnu.so"

# Check if the target file (lib_cxx.so) does NOT exist
if [ ! -f "$TARGET2" ]; then
    echo "Copying: $SOURCE2 --> $TARGET2"
    cp "$SOURCE2" "$TARGET2"
else
    # If the file exists, notify the user and skip the linking step.
    echo "$TARGET2 already exists. Skipping."
fi

# To check license
simulation/anygrasp_sdk/license_registration/license_checker -c simulation/anygrasp_sdk/grasp_detection/license/licenseCfg.json