Firstly you need to prepare the dataset and pre-trained models as described [here](https://github.com/Nicholasli1995/EgoNet/blob/master/docs/preparation.md).

Then modify the directories by

```bash
cd ${EgoNet_DIR}/configs && vim KITTI_inference:demo.yml
```

Edit dirs:ckpt to your pre-trained model directory.

Edit dataset:root to your KITTI directory.

Finally, go to ${EgoNet_DIR}/tools and run

```bash
 python inference.py --cfg "../configs/KITTI_inference:demo.yml" --visualize True --batch_to_show 2
```

You can set --batch_to_show to other integers to see more results.
