## Data Preparation 
You need to download KITTI dataset [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Download left images, calibration files and labels.
Download the split files here and place them at ${YOUR_KITTI_DIR/SPLIT/ImageSets}.
Your data folder should look like this:

   ```
   ${YOUR_KITTI_DIR}
   ├── training
      ├── calib
          ├── xxxxxx.txt (Camera parameters for image xxxxxx)
      ├── image_2
          ├── xxxxxx.png (image xxxxxx)
      ├── label_2
          ├── xxxxxx.txt (object labels for image xxxxxx)
      ├── ImageSets
         ├── train.txt
         ├── val.txt   
         ├── trainval.txt        
   ├── testing
      ├── calib
          ├── xxxxxx.txt (Camera parameters for image xxxxxx)
      ├── image_2
          ├── xxxxxx.png (image xxxxxx)
      ├── ImageSets
         ├── test.txt
   ```
    
## Environment
You need to create an environment that meets the following dependencies.

- Python 3
- Numpy 
- PyTorch (GPU required)
- Scipy
- Matplotlib
- OpenCV
- PyYAML

To refer to an tested environment, see [spec-list.txt](test). Other versions may work but are not tested.
The recommended environment manager is Anaconda, which can create an environment using this provided spec-list. 
