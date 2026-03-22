# MAE/ECE 148 YOLO Labeler, Training, and State Estimation
This is part of Team 1's project for the MAE/ECE 148 project in Winter 2026. See below repositories for other parts:
ROS2 robocar tracking using the model trained on this repository: https://github.com/perilune42/ros2_tracking/tree/main
F1TENTH gym environment ROS2 communication bridge: https://github.com/arushbisht12/autonomous_racing_sim

The `yolo_labeler` directory contains an easy-to-use GUI to quickly create YOLO-compatible labels, and the `yolo_detection` directory contains the tools needed to train a PyTorch model using Ultralytics and to run a Luxonis OAK-D camera. The `opponent_ekf` directory contains Kalman filter scripts that are able to output a robust estimation of a detected car's state.

This project was intended to be used to recognize other robocars for the class, though it can easily be used for recognizing other object classes if desired.

See the README in each directory for more detailed instructions.
