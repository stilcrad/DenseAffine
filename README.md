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
Download the weights at https://pan.baidu.com/s/1EdsAqKJ1HKVS5uLPMAxDSQ?pwd=idw2 password: idw2 or [https://drive.google.com/file/d/1W82QJ5lgrsVQql30k_NZWZE5YH9lhgaE/view?usp=drive_link](https://drive.google.com/file/d/1W82QJ5lgrsVQql30k_NZWZE5YH9lhgaE/view?usp=drive_link).  Put it in weights folder.


```shell

python demo/affine_feature_estimator.py
```
The matching results of the images are saved in the results folder.

Download the KITTI sequens 04 for relative pose estimation. The URL of the dataset is https://pan.baidu.com/s/1EdsAqKJ1HKVS5uLPMAxDSQ?pwd=idw2 password: idw2 or [https://drive.google.com/file/d/1W82QJ5lgrsVQql30k_NZWZE5YH9lhgaE/view?usp=drive_link](https://drive.google.com/file/d/1W82QJ5lgrsVQql30k_NZWZE5YH9lhgaE/view?usp=drive_link).
```shell

python demo/affine_feature_relative_pose.py
```


## Acknowledgements

We have used code and been inspired by https://github.com/Parskatt/dkm, https://github.com/laoyandujiang/S3Esti, and https://github.com/ducha-aiki/affnet, https://github.com/DagnyT/hardnet, https://github.com/Reagan1311/LOCATE, https://github.com/danini/graph-cut-ransac. Sincere thanks to these authors for their codes.
