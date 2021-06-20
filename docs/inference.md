## Reproduce D4LCN + EgoNet
Firstly you need to prepare the dataset and pre-trained models as described [here](https://github.com/Nicholasli1995/EgoNet/blob/master/docs/preparation.md).

Then modify the directories by

```bash
cd ${EgoNet_DIR}/configs && vim KITTI_inference:demo.yml
```

Edit dirs:ckpt to your pre-trained model directory.

Edit dataset:root to your KITTI directory.

Finally, go to ${EgoNet_DIR}/tools and run

```bash
 python inference.py --cfg "../configs/KITTI_inference:demo.yml"
```

This will load D4LCN predictions, refine their vehicle orientation predictions and save the results.
The official evaluation program will automatically run to produce quantitative performance.
