#!/usr/bin/env python3
"""
YOLO Video Object Detection with Bounding Box Overlay
------------------------------------------------------
Usage:
    python yolo_video_detect.py --model yolov8n.pt --input video.mp4 --output output.mp4

Requirements:
    pip install ultralytics opencv-python
"""

import argparse
import sys
import time
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO object detection on a video and overlay bounding boxes."
    )
    parser.add_argument(
        "--model", required=True, help="Path to YOLO .pt model file (e.g. yolov8n.pt)"
    )
    parser.add_argument(
        "--input", required=True, help="Path to input video file"
    )
    parser.add_argument(
        "--output", default="output.mp4", help="Path to output video file (default: output.mp4)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold for detections (default: 0.25)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="IoU threshold for NMS (default: 0.45)"
    )
    parser.add_argument(
        "--classes", nargs="+", type=int, default=None,
        help="Filter by class IDs, e.g. --classes 0 2 (person, car)"
    )
    parser.add_argument(
        "--thickness", type=int, default=2,
        help="Bounding box line thickness (default: 2)"
    )
    parser.add_argument(
        "--font-scale", type=float, default=0.6,
        help="Font scale for labels (default: 0.6)"
    )
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Hide class labels and confidence scores"
    )
    parser.add_argument(
        "--device", default=None,
        help="Device to run on: 'cpu', '0' (GPU 0), etc. Auto-detected if not set."
    )
    parser.add_argument(
        "--predict-every", type=int, default=10, metavar="N",
        help="Run YOLO inference only once every N frames; reuse boxes on skipped frames (default: 10)"
    )
    return parser.parse_args()


# ── Colour palette ──────────────────────────────────────────────────────────
PALETTE = [
    (255,  56,  56), (255, 157,  151), (255, 112,  31), (255, 178,  29),
    (207, 210,  49), ( 72, 249,  10), (146, 204,  23), ( 61, 219, 134),
    ( 26, 147,  52), (  0, 212, 187), ( 44, 153, 168), (  0, 194, 255),
    ( 52, 121, 246), (  0,   9, 246), (130,  49, 247), (184,  55, 247),
    (255,   9, 243), (255, 129, 208), (209,  97,   0), (  0, 163, 117),
]

def colour_for(class_id: int):
    return PALETTE[class_id % len(PALETTE)]


def draw_detections(frame, boxes, names, show_labels: bool, thickness: int, font_scale: float):
    """Draw bounding boxes (and optional labels) onto *frame* in-place."""
    for box in boxes:
        cls_id  = int(box.cls[0])
        conf    = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        colour  = colour_for(cls_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

        if show_labels:
            label = f"{names[cls_id]} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            # filled background rectangle for readability
            top = max(y1 - th - baseline - 4, 0)
            cv2.rectangle(frame, (x1, top), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(
                frame, label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA
            )
    return frame


def main():
    args = parse_args()

    # ── Validate inputs ──────────────────────────────────────────────────────
    model_path = Path(args.model)
    input_path = Path(args.input)

    if not model_path.exists():
        sys.exit(f"[ERROR] Model not found: {model_path}")
    if not input_path.exists():
        sys.exit(f"[ERROR] Input video not found: {input_path}")

    # ── Load YOLO model ──────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[ERROR] ultralytics not installed. Run: pip install ultralytics")

    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(str(model_path))
    if args.device is not None:
        model.to(args.device)

    # ── Open video ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {input_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video: {width}x{height} @ {fps:.2f} fps | {total} frames")

    # ── Set up writer ────────────────────────────────────────────────────────
    output_path = Path(args.output)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        sys.exit(f"[ERROR] Cannot open output writer: {output_path}")

    # ── Process frames ───────────────────────────────────────────────────────
    show_labels = not args.no_labels
    frame_idx   = 0
    t0          = time.time()

    print("[INFO] Processing…  (Ctrl-C to abort)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(
                source     = frame,
                conf       = args.conf,
                iou        = args.iou,
                classes    = args.classes,
                verbose    = False,
            )

            for result in results:
                if result.boxes is not None and len(result.boxes):
                    draw_detections(
                        frame,
                        result.boxes,
                        result.names,
                        show_labels,
                        args.thickness,
                        args.font_scale,
                    )

            writer.write(frame)
            frame_idx += 1

            # Progress indicator every 30 frames
            if frame_idx % 30 == 0:
                elapsed = time.time() - t0
                fps_proc = frame_idx / elapsed
                pct = (frame_idx / total * 100) if total > 0 else 0
                print(
                    f"\r  Frame {frame_idx:>6}/{total}  ({pct:5.1f}%)  "
                    f"{fps_proc:5.1f} fps",
                    end="", flush=True
                )

    except KeyboardInterrupt:
        print("\n[INFO] Aborted by user.")

    # ── Cleanup ──────────────────────────────────────────────────────────────
    cap.release()
    writer.release()

    elapsed = time.time() - t0
    print(f"\n[INFO] Done. {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx/elapsed:.1f} fps avg)")
    print(f"[INFO] Output saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()