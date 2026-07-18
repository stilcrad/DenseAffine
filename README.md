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
Download the weights at https://pan.baidu.com/s/1EdsAqKJ1HKVS5uLPMAxDSQ?pwd=idw2 password: idw2 or [https://drive.google.com/file/d/1WX6Ug0ERmT9JC5OwMVWav20RBhkKOsvr/view?usp=sharing])).  Put it in weights folder.

We have updated the file of weights, and it may be necessary to make slight modifications to the loaded code when using it. We will open source the training code after the expansion paper is accepted.


```shell

python demo/affine_feature_estimator.py
```
The matching results of the images are saved in the results folder.

Download the KITTI sequens 04 for relative pose estimation. The URL of the dataset is https://pan.baidu.com/s/1EdsAqKJ1HKVS5uLPMAxDSQ?pwd=idw2 password: idw2 or [https://drive.google.com/file/d/1W82QJ5lgrsVQql30k_NZWZE5YH9lhgaE/view?usp=drive_link](https://drive.google.com/file/d/1HLhSX17aGZ9SHRwVfNbtjeXL43Vqnjjf/view?usp=drive_link))
.
```shell

python demo/affine_feature_relative_pose.py
```


## Acknowledgements

We have used code and been inspired by https://github.com/Parskatt/dkm, https://github.com/laoyandujiang/S3Esti, and https://github.com/ducha-aiki/affnet, https://github.com/DagnyT/hardnet, https://github.com/Reagan1311/LOCATE, https://github.com/danini/graph-cut-ransac. Sincere thanks to these authors for their codes.

## cite
If you find this work useful, please cite:
```bibtex
@inproceedings{sun2025learning,
  title={Learning affine correspondences by integrating geometric constraints},
  author={Sun, Pengju and Guan, Banglei and Yu, Zhenbao and Shang, Yang and Yu, Qifeng and Barath, Daniel},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={27038--27048},
  year={2025}
}
