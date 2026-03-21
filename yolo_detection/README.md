# YOLO training and running on OAK-D camera 

This directory contains scripts to train and run a YOLO detection model.

Using Python 3.12 is recommended.
Requirements: ultralytics, torch, torchvision

The robocar dataset is not included in this repository due to large file sizes. It is available on request by contacting kamei@ucsd.edu

1. Populate the dataset folder with labeled images.

2. Change the `data.yaml` file to detect different classes if desired.

2. Enter the yolo_training.ipynb notebook.

3. If applicable, verify that pytorch is detecting a GPU.

4. imgsz is the square image size and should be a multiple of 32, keeping it as 256 is recommended. Adjust the number of epochs and other training parameters if desired, and start training.

5. PyTorch model will be saved in the current directory. Check the generated runs/ directory for some useful metrics.

6. (Optional) To test the model on an input video, run `python yolo_video_detect.py --model [MODEL PATH] --input [VIDEO PATH]` and see if objects are identified correctly.

The following sections are for running the model on a Luxonis OAK-D camera after a .pt model has been trained to desired performance.

1. Go to https://hub.luxonis.com/ai/tools to convert the model into an archive usable for depthai devices. \
It will require an account, simply make one for free. Upload the model and set the correct image shape (MUST be square and a multiple of 32 in each dimension). \
Download the NN archive after conversion and place in this directory.

2. Run `pip install depthai==3.3.0`. Other versions are not guaranteed to work.

3. Plug in the camera to your device. If running on a Raspberry Pi or a similar SBC, you may get error failing to boot the camera if it does not receive enough power. Using an externally powered USB hub is recommended.

4. To run simple object detection on the camera, set the MODEL_PATH in yolo_oakd_detection.py and run it. 

5. To run object tracking with spatial coordinates output, set the MODEL_PATH in yolo_oakd_tracking.py and run it.  


