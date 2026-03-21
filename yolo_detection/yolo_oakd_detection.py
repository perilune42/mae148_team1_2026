from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time


MODEL_PATH = "model.rvc2.tar.xz"
CAPTURE_PATH = "oakd_imgs"

import re
from pathlib import Path

save_dir = Path(CAPTURE_PATH)
save_dir.mkdir(exist_ok=True)

pattern = re.compile(r"frame_(\d+)\.png")

existing_indices = []

for f in save_dir.glob("frame_*.png"):
    m = pattern.match(f.name)
    if m:
        existing_indices.append(int(m.group(1)))

img_idx = max(existing_indices) + 1 if existing_indices else 0
print(f"Starting from frame_{img_idx:04d}")

with dai.Pipeline() as pipeline:

    cameraNode = pipeline.create(dai.node.Camera).build()
    detectionNetwork = pipeline.create(dai.node.DetectionNetwork)
    output = cameraNode.requestOutput((256, 256), dai.ImgFrame.Type.RGB888p, dai.ImgResizeMode.LETTERBOX, 30)
    output.link(detectionNetwork.input)
    detectionNetwork.setNNArchive(dai.NNArchive(Path(MODEL_PATH)))
    labelMap = detectionNetwork.getClasses()
    qRgb = detectionNetwork.passthrough.createOutputQueue()
    qDet = detectionNetwork.out.createOutputQueue()
    pipeline.start()
    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)


    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        color = (255, 0, 0)
        for detection in detections:
            bbox = frameNorm(
                frame,
                (detection.xmin, detection.ymin, detection.xmax, detection.ymax),
            )
            cv2.putText(
                frame,
                labelMap[detection.label],
                (bbox[0] + 10, bbox[1] + 20),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            cv2.putText(
                frame,
                f"{int(detection.confidence * 100)}%",
                (bbox[0] + 10, bbox[1] + 40),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # Show the frame
        cv2.imshow(name, frame)

    while pipeline.isRunning():
        inRgb: dai.ImgFrame = qRgb.get()
        inDet: dai.ImgDetections = qDet.get()
        if inRgb is not None:
            frame = inRgb.getCvFrame()
            # Keep a copy of the raw frame for saving
            raw_frame = frame.copy()

            cv2.putText(
                frame,
                "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                (2, frame.shape[0] - 4),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.4,
                color2,
            )

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        if frame is not None:
            displayFrame("rgb", frame)
            print("FPS: {:.2f}".format(counter / (time.monotonic() - startTime)))

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            pipeline.stop()
            break
        elif key == ord("s"):  # Save the raw frame
            save_path = f"{CAPTURE_PATH}/frame_{img_idx:04d}.png"
            print(f"Saving to: {save_path}")
            cv2.imwrite(str(save_path), raw_frame)
            img_idx += 1