#!/bin/sh
pip uninstall -y scikit-learn pandas
pip install pandas scikit-learn
pip uninstall -y scikit-learn
pip install scikit-learn==1.0.2
pip install -U numpy
pip uninstall -y numpy
pip install numpy==1.26.4
pip install floris==4.2.2
curl -sS https://starship.rs/install.sh | sh
SITE_PACKAGES=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "eval $(starship init bash)" >> ~/.bashrc
mkdir -p $SITE_PACKAGES/Cython-3.0.10.dist-info/
touch $SITE_PACKAGES/Cython-3.0.10.dist-info/METADATA
# pip leaves a stale cython-<version>.dist-info without METADATA behind, which breaks later pip installs
for d in $SITE_PACKAGES/[cC]ython-*.dist-info; do [ -f "$d/METADATA" ] || touch "$d/METADATA"; done
pip install hypermapper==2.2.13

# fix torch issues
apt-get update && apt-get install -y patchelf
patchelf --clear-execstack $SITE_PACKAGES/torch/lib/libtorch_cpu.so

# fix charset issues
find $SITE_PACKAGES/ -name "*charset_normalizer*" -exec rm -rf {} +
pip install --no-cache-dir charset-normalizer

# fix setuptools issues
pip install "setuptools<70.0.0"

# prepare data and files
python env_setup.py
bash