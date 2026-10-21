#!/bin/sh
pip uninstall -y scikit-learn pandas
pip install pandas scikit-learn
pip uninstall -y scikit-learn
pip install scikit-learn==1.0.2
pip install -U numpy
pip uninstall -y numpy
pip install numpy==1.24.4
pip install floris==4.2.2
curl -sS https://starship.rs/install.sh | sh
echo "eval $(starship init bash)" >> ~/.bashrc
mkdir -p /usr/local/lib/python3.9/site-packages/Cython-3.0.10.dist-info/
touch /usr/local/lib/python3.9/site-packages/Cython-3.0.10.dist-info/METADATA
touch /usr/local/lib/python3.9/site-packages/cython-3.2.4.dist-info/METADATA
pip install hypermapper

# fix torch issues
apt-get update && apt-get install -y patchelf
patchelf --clear-execstack /usr/local/lib/python3.9/site-packages/torch/lib/libtorch_cpu.so

# fix charset issues
find /usr/local/lib/python3.9/site-packages/ -name "*charset_normalizer*" -exec rm -rf {} +
pip install --no-cache-dir charset-normalizer

# fix setuptools issues
pip install "setuptools<70.0.0"

# prepare data and files
python env_setup.py
bash