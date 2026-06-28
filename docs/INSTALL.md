# Installation

WHAM has been implemented and tested on Ubuntu 20.04 and 22.04 with python = 3.9. We provide [anaconda](https://www.anaconda.com/) environment to run WHAM as below.

```bash
#Install gcc-9
sudo apt install gcc-9 g++-9 -y


# Clone the repo
git clone https://github.com/yohanshin/WHAM.git --recursive
cd WHAM/

# Create Conda environment
conda create -n wham python=3.10
conda activate wham

# Install PyTorch libraries and pytorch3d
conda install -c conda-forge   cudatoolkit=11.3  cudnn -y
conda install -c nvidia cuda-nvcc=11.3 -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c conda-forge mkl=2024.0 -y

pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 torch_scatter==2.0.9  -f https://download.pytorch.org/whl/torch/ -f https://download.pytorch.org/whl/torchaudio/ -f https://download.pytorch.org/whl/torchvision/ -f https://data.pyg.org/whl/torch-2.0.1%2Bcu117.html
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt201/download.html
pip install numpy==1.22.3
pip install "setuptools<81"
pip list --format=freeze > constraints.txt


# Install ViTPose
pip install -v -e third-party/ViTPose --no-build-isolation -c constraints.txt

# Install DPVO
cd third-party/DPVO
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty && rm -rf eigen-3.4.0.zip

cp -r /usr/local/cuda/include/* $CONDA_PREFIX/include/
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export TORCH_CUDA_ARCH_LIST="7.5"
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9

rm -rf build/
pip install . --no-build-isolation
# Install WHAM dependencies
cd ../..
pip install -r requirements.txt --no-build-isolation
```
