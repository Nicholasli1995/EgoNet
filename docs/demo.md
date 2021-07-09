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

The visualized 3D bounding boxes are distinguished by their colors: 
1. Black indicates ground truth 3D boxes.
2. Magenta indicates 3D bounding boxes predicted by another 3D object detector ([D4LCN](https://github.com/dingmyu/D4LCN)).
3. Red indicates the predictions of Ego-Net, using the 2D bounding boxes from [D4LCN](https://github.com/dingmyu/D4LCN).
4. Yellow indicates the predictions of Ego-Net, using the ground truth 2D bounding boxes.
