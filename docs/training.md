Firstly you need to prepare the dataset as described [here](https://github.com/Nicholasli1995/EgoNet/blob/master/docs/preparation.md).

Then download a start point model [here](https://drive.google.com/file/d/1VFtMGgBG0cLGnbr3brrnPnJii2xGYj-9/view?usp=sharing) and place it at ${EgoNet_DIR}/resources. 

The training phase consists of two stages which are described as follows. 

For training on other datasets. You need to prepare the training images and camera parameters accordingly.

## Stage 1: train a lifter (L.pth)
You need to modify the configuration by

```bash
cd ${EgoNet_DIR}/configs && vim KITTI_train_lifting.yml
```
Edit dataset:root to your KITTI directory.

(Optional) Edit dirs:output to where you want to save the output model.

(Optional) You can evaluate during training by setting eval_during to True.

Finally, run

```bash
 cd tools
 python train_lifting.py --cfg "../configs/KITTI_train_lifting.yml"
```


## Stage 2: train the remaining part (HC.pth)
You need to modify the configuration by

```bash
cd ${EgoNet_DIR}/configs && vim KITTI_train_IGRs.yml
```

Edit dataset:root to your KITTI directory.

Edit gpu_id according to your local machine and set batch_size based on how much GPU memory you have. 

(Optional) Edit dirs:output to where you want to save the output model.

(Optional) You can evaluate during training by setting eval_during to True.

(Optional) Edit ss to enable self-supervised representation learning. You need to prepare unlabeled ApolloScape images and download record [here](https://drive.google.com/file/d/1uPdOC7LioomMF5DieUNrx3aZKsgobP5U/view?usp=sharing).

(Optional) Edit training_settings:debug to disable saveing intermediate training results.

Finally, run

```bash
 cd tools
 python train_IGRs.py --cfg "../configs/KITTI_train_IGRs.yml"
```
