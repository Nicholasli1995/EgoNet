[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/exploring-intermediate-representation-for/vehicle-pose-estimation-on-kitti-cars-hard)](https://paperswithcode.com/sota/vehicle-pose-estimation-on-kitti-cars-hard?p=exploring-intermediate-representation-for)
# EgoNet
Official project website for the CVPR 2021 paper "Exploring intermediate representation for monocular vehicle pose estimation"

This repository is under preparation. Inference code and pre-trained models are coming soon for quick deployment. Training code will be released before the main conference.

<p align="center">
  <img src="https://github.com/Nicholasli1995/EgoNet/blob/master/imgs/teaser.jpg"  width="830" height="200" />
</p>

## Performance: AP<sup>BEV</sup>@R<sub>40</sub> on KITTI val set (monocular RGB)

| Method                    | Reference|Easy|Moderate|Hard|
| ------------------------- | ---------------| --------------| --------------| --------------| 
|[M3D-RPN](https://arxiv.org/abs/1907.06038)|ICCV 2019|20.85| 15.62| 11.88|
|[MonoDIS](https://openaccess.thecvf.com/content_ICCV_2019/papers/Simonelli_Disentangling_Monocular_3D_Object_Detection_ICCV_2019_paper.pdf)|ICCV 2019|18.45 |12.58 |10.66|
|[MonoPair](https://arxiv.org/abs/2003.00504)|CVPR 2020|24.12| 18.17| 15.76|
|[D4LCN](https://github.com/dingmyu/D4LCN)|CVPR 2020|31.53 |22.58  |17.87|
|[Kinematic3D](https://arxiv.org/abs/2007.09548)|ECCV 2020|27.83| 19.72| 15.10|
|[GrooMeD-NMS](https://github.com/abhi1kumar/groomed_nms)|CVPR 2021 |27.38|19.75|15.92|
|[MonoDLE](https://github.com/xinzhuma/monodle)|CVPR 2021|24.97| 19.33| 17.01|
|Ours           |CVPR 2021 |**33.60**|**25.38**|**22.80**|

## Reference

    @InProceedings{Li_2021_CVPR,
    author = {Li, Shichao and Yan, Zengqiang and Li, Hongyang and Cheng, Kwang-Ting},
    title = {Exploring intermediate representation for monocular vehicle pose estimation},
    booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2021}
    }
