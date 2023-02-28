# Argonne CL

## Install

- CUDA

```
# run this in Chameleon node
sudo apt-get remove --purge '^nvidia-.*'
sudo apt-get remove --purge '^libnvidia-.*'
sudo apt-get remove --purge '^cuda-.*'
sudo apt-get install linux-headers-$(uname -r)
```

Next, we need to download the latest/specific version of CUDA (11.7)
https://developer.nvidia.com/cuda-11-7-0-download-archive

- Dependencies

```
conda create -n cl python=3.10 pip
conda activate cl
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install git+https://github.com/ContinualAI/avalanche.git@0088c3092af918ac2c16d3f945be8dd62415a01c
conda install pandas black matplotlib scikit-learn numpy -c conda-forge
```
