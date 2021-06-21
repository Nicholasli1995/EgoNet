Firstly you need to prepare the dataset and pre-trained models as described [here](https://github.com/Nicholasli1995/EgoNet/blob/master/docs/preparation.md).

## Reproduce D4LCN + EgoNet on the val split
You need to modify the directories by

```bash
cd ${EgoNet_DIR}/configs && vim KITTI_inference:demo.yml
```
Edit dirs:output to where you want to save the predictions.

Edit dirs:ckpt to your pre-trained model directory.

Edit dataset:root to your KITTI directory.

Finally, go to ${EgoNet_DIR}/tools and run

```bash
 python inference.py --cfg "../configs/KITTI_inference:demo.yml"
```

This will load D4LCN predictions, refine their vehicle orientation predictions and save the results.
The official evaluation program will automatically run to produce quantitative performance.

## Reproduce results on the test split
You need to modify the directories by

```bash
cd ${EgoNet_DIR}/configs && vim KITTI_inference:test_submission.yml
```
Edit dirs:output to where you want to save the predictions.

Edit dirs:ckpt to your pre-trained model directory.

Edit dataset:root to your KITTI directory.

Finally, go to ${EgoNet_DIR}/tools and run

```bash
 python inference.py --cfg "../configs/KITTI_inference:test_submission.yml"
```

This will load prepared 2D bounding boxes, predict the vehicle orientation and save the predictions.

Now you can zip the results and submit it to the [official server](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)!

You can hit [91.23% AOS](http://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=e5233225fd5ef36fa63eb00252d9c00024961f2c) for the moderate setting! This is the **most important** metric for joint vehicle detection and pose estimation on KITTI. You achieved this with a single RGB image without extra training data.
