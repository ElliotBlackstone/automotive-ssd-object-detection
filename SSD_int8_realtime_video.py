import argparse
import time
from dataclasses import replace

import cv2
import numpy as np

from SSDInt8_ONNX_Pred import SSDInt8ONNXPredictor, PreprocessConfig


# to run:
# desktop
# python SSD_int8_realtime_video.py --model C:\Users\eblac\Documents\GitHub\self-driving-car\PTQ_testing\ssd_int8_with_pre_post.onnx --show-fps

# laptop
# python SSD_int8_realtime_video.py --model C:\Users\eblac\OneDrive\Documents\GitHub\self-driving-car\PTQ_testing\ssd_int8_with_pre_post.onnx --show-fps --camera 1


def build_class_names_fg(class_to_idx: dict) -> list[str]:
    """
    Foreground class names in index order 0..C-2.
    Your model's conf has C=6 => foreground count is 5, matching your dict.
    """
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    n = len(idx_to_class)
    return [idx_to_class[i] for i in range(n)]


def draw_predictions_bgr(
    frame_bgr: np.ndarray,
    pred: dict,
    show_labels: bool = True,
    score_fmt: str = "{:.2f}",
) -> np.ndarray:
    """
    pred = {"labels": [str], "scores": [float], "boxes": np.ndarray (K,4) xyxy in frame pixels}
    """
    out = frame_bgr #.copy()
    H, W = out.shape[:2]

    boxes = pred["boxes"]
    labels = pred["labels"]
    scores = pred["scores"]

    # Defensive: handle empty
    if boxes is None or len(labels) == 0:
        return out

    boxes = np.asarray(boxes, dtype=np.float32)

    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i]
        x1 = int(np.clip(x1, 0, W - 1))
        x2 = int(np.clip(x2, 0, W - 1))
        y1 = int(np.clip(y1, 0, H - 1))
        y2 = int(np.clip(y2, 0, H - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if show_labels:
            name = labels[i]
            sc = scores[i]
            txt = f"{name}:{score_fmt.format(sc)}"
            cv2.putText(
                out,
                txt,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=str, help="Path to INT8 ONNX model (ssd_int8.onnx)")
    ap.add_argument("--camera", default=0, type=int, help="Webcam index")
    ap.add_argument("--width", default=1280, type=int)
    ap.add_argument("--height", default=720, type=int)
    ap.add_argument("--fps", default=30, type=int)
    ap.add_argument("--score-thresh", default=0.20, type=float)
    ap.add_argument("--nms-thresh", default=0.50, type=float)
    ap.add_argument("--max-per-img", default=100, type=int)
    ap.add_argument("--class-agnostic", action="store_true")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--show-fps", action="store_true")
    args = ap.parse_args()


    class_to_idx = {"biker": 0, "car": 1, "pedestrian": 2, "trafficLight": 3, "truck": 4}

    predictor = SSDInt8ONNXPredictor(
        onnx_model_path=args.model,
        class_to_idx=class_to_idx,
        providers=["CPUExecutionProvider"],
        preprocess_cfg=PreprocessConfig(input_color="bgr"),
    )


    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    # Reduce capture latency / buffering
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    cap.set(cv2.CAP_PROP_FPS, float(args.fps))

    # Ask camera for MJPG (many Logitech cams support; helps throughput)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    # Warm-up (first frame + first run)
    ok, frame_bgr = cap.read()
    if not ok:
        raise RuntimeError("Failed to read initial frame.")
    _ = predictor(frame_bgr)



    # ---- video recording setup ----
    writer = None
    out_path = "ssd_int8_demo.mp4"
    fourcc_out = cv2.VideoWriter_fourcc(*"mp4v")
    H0, W0 = frame_bgr.shape[:2]
    writer = cv2.VideoWriter(out_path, fourcc_out, float(args.fps), (W0, H0))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed to open. Try XVID/AVI or MJPG/AVI.")

    fps_smoothed = 0.0
    last_print = time.perf_counter()

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            t0 = time.perf_counter()

            # Predictor expects RGB uint8 HWC
            # frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pred = predictor(frame_bgr)  # {"labels":..., "scores":..., "boxes":...} in frame pixel coords

            vis = draw_predictions_bgr(frame_bgr, pred, show_labels=not args.no_labels)

            dt = time.perf_counter() - t0
            inst_fps = (1.0 / dt) if dt > 0 else 0.0
            fps_smoothed = 0.9 * fps_smoothed + 0.1 * inst_fps

            if args.show_fps:
                cv2.putText(
                    vis,
                    f"FPS: {fps_smoothed:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )

            if writer is not None:
                writer.write(vis)

            cv2.imshow("SSD INT8 ONNX Runtime", vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break

            now = time.perf_counter()
            if now - last_print > 5.0:
                print(f"[info] smoothed FPS ~ {fps_smoothed:.1f} | dets={len(pred['labels'])}")
                last_print = now

    finally:
        if writer is not None:
            writer.release()
            print(f"Saved video: {out_path}")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
