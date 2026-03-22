# MAE 148 YOLO Speed Labeler

This tool allows us to label our robocar data quickly and export it for the class dataset.
# Taking Images

NOTE: Take photos at varying distances away (about 1-10 meters) away from the robocar in different environments (EBU2 track, anywhere outside)

1. Take pictures in landscape orientation at any aspect ratio

2. Take photos on the same level as the robocar (all angles)

# Image Setup

1. If the images are in .heic format, convert them to .jpg or .png first as it is not supported

2. Move all images into a single directory

# Labeling Setup

1. Clone this repo

2. Run 'pip install -r requirements.txt' in a venv or 'uv sync' if you have the uv package manager

3. Enter the yolo_labeler folder and launch gui with 'python3 label_gui.py' or 'uv run label_gui.py'

4. Enter team number, press CMD + O to open folder

5. Label each image. Note that previously applied labels will not show up, but making any new labels will overwrite the old labels.

6. Export the dataset folder and copy it into the yolo_detection directory to prepare for training.

