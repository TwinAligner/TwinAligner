uv pip install --no-build-isolation -r simulation/requirements_fp.txt
export CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/pybind11/share/cmake/pybind11:$CONDA_PREFIX/include/eigen3
cd simulation/FoundationPose++/FoundationPose
bash build_all_conda.sh
cd ../Cutie
uv pip install -e .
cd ../../..