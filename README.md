# Encoder-Decoder Based Convolutional Neural Networks with Multi-Scale-Aware Modules for Crowd Counting（ICPR 2020）
##  Official Implementation of "Encoder-Decoder Based Convolutional Neural Networks with Multi-Scale-Aware Modules for Crowd Counting" [LINK](https://arxiv.org/abs/2003.05586)
Many thanks to [BL](https://github.com/ZhihengCV/Bayesian-Crowd-Counting), [SFANet](https://github.com/pxq0312/SFANet-crowd-counting/) and [CAN](https://github.com/weizheliu/Context-Aware-Crowd-Counting) for their useful publications and repositories.

For complete UCF-QNRF and Shanghaitech training code, please refer to [BL](https://github.com/ZhihengCV/Bayesian-Crowd-Counting) and [SFANet](https://github.com/pxq0312/SFANet-crowd-counting/) respectively.

Please see ```models``` for our M-SFANet and M-SegNet implementations.

## Density maps Visualization

![](images/img_0071_heatpmap.png)
![](images/seg_img_0323_heatpmap.png)

## Citation
If you find the code useful for your research, please cite our paper:

```
@inproceedings{thanasutives2021encoder,
  title={Encoder-Decoder Based Convolutional Neural Networks with Multi-Scale-Aware Modules for Crowd Counting},
  author={Thanasutives, Pongpisit and Fukui, Ken-ichi and Numao, Masayuki and Kijsirikul, Boonserm},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  pages={2382--2389},
  year={2021},
  organization={IEEE}
}
```
### Datasets (NEW)
To reproduce the results reported in the paper, you may use these preprocessed datasets. This is not completed yet, and might be updated in the future.

Shanghaitech B dataset that is preprocessed using the Gaussian kernel [Link](https://drive.google.com/file/d/1Jjmvp-BEa-_81rXgX1bvdqi5gzteRdJA/view?usp=sharing)

Bayesian preprocessed (following [BL](https://github.com/ZhihengCV/Bayesian-Crowd-Counting)) Shanghaitech datasets (A&B) [Link](https://drive.google.com/file/d/1azoaoRGxfXI7EkSXGm4RrX18sBnDxUtP/view?usp=sharing)

The Beijing-BRT dataset [Link](https://drive.google.com/file/d/1JRjdMWtWiLxocHensFfJzqLoJEFksjVy/view?usp=sharing) (Originally from [BRT](https://github.com/XMU-smartdsp/Beijing-BRT-dataset))

### Pretrained Weights
Shanghaitech A&B [Link](https://drive.google.com/file/d/1MxGZjapIv6O-hzxEeHY7c93723mhGKrG/view?usp=sharing)

To test the visualization code you should use the pretrained M_SegNet* on UCF_QNRF [Link](https://drive.google.com/file/d/1fGuH4o0hKbgdP1kaj9rbjX2HUL1IH0oo/view?usp=sharing) (The pretrained weights of M_SFANet* are also included.)
