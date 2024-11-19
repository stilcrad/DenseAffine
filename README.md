## Requirements
- CUDA 12.0
- Python 3.6 (or later)
- torch 2.1.0
- torchvision 0.16.0
- opencv_python 4.10.0.84
- kornia 0.7.0
## Data preparation



## Install 

```shell
# create virtual environment

conda create -n DenseAffine python=3.10

conda activate DenseAffine

# install DenseAffine requirements

cd DenseAffine

pip install -r requirements.txt

```

## Get start


```shell

python test.py 

```
## Acknowledgements

We have used code and been inspired by https://github.com/Parskatt/dkm, https://github.com/laoyandujiang/S3Esti, and https://github.com/ducha-aiki/affnet, https://github.com/DagnyT/hardnet, https://github.com/Reagan1311/LOCATE. Sincere thanks to these authors for their codes.
