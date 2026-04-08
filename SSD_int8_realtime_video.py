import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import platform

from SSDInt8_ONNX_Pred import SSDInt8ONNXPredictor, PreprocessConfig


# to run:
# desktop
# python SSD_int8_realtime_video.py --model C:\Users\eblac\Documents\GitHub\self-driving-car\PTQ_testing\ssd_int8_with_pre_post.onnx --show-fps
#
# laptop
# python SSD_int8_realtime_video.py --model C:\Users\eblac\OneDrive\Documents\GitHub\self-driving-car\PTQ_testing\ssd_int8_with_pre_post.onnx --show-fps --camera 1


def draw_predictions_bgr(
    frame_bgr: np.ndarray,
    pred: dict,
    show_labels: bool = True,
    score_fmt: str = "{:.2f}",
) -> np.ndarray:
    """
    pred = {"labels": [str], "scores": [float], "boxes": np.ndarray (K,4) xyxy in frame pixels}
    """
    out = frame_bgr
    H, W = out.shape[:2]

    boxes = pred["boxes"]
    labels = pred["labels"]
    scores = pred["scores"]

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
            txt = f"{labels[i]}:{score_fmt.format(scores[i])}"
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


def open_camera(source, backend: str):
    backend_map = {
        "any": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "v4l2": cv2.CAP_V4L2,
        "gstreamer": cv2.CAP_GSTREAMER,
    }

    sysname = platform.system().lower()

    if backend != "auto":
        trial = [backend]
    else:
        if sysname == "windows":
            trial = ["dshow", "msmf", "any"]
        elif sysname == "linux":
            trial = ["v4l2", "gstreamer", "any"]
        else:
            trial = ["any"]

    last_err = None
    for b in trial:
        cap = cv2.VideoCapture(source, backend_map[b])
        if cap.isOpened():
            return cap, b
        last_err = b
        cap.release()

    raise RuntimeError(
        f"Could not open camera source={source!r} with backends={trial} (last tried: {last_err})"
    )


def open_video_writer(out_path: str, fps: float, frame_size: tuple[int, int]):
    """Open a writer. Prefer mp4v for .mp4, otherwise fall back to XVID .avi."""
    out_path = str(out_path)
    suffix = Path(out_path).suffix.lower()

    candidates = []
    if suffix == ".mp4":
        candidates.append((out_path, cv2.VideoWriter_fourcc(*"mp4v")))
    candidates.append((out_path, cv2.VideoWriter_fourcc(*"XVID")))
    candidates.append((str(Path(out_path).with_suffix(".avi")), cv2.VideoWriter_fourcc(*"XVID")))

    for candidate_path, fourcc in candidates:
        writer = cv2.VideoWriter(candidate_path, fourcc, float(fps), frame_size)
        if writer.isOpened():
            return writer, candidate_path
        writer.release()

    raise RuntimeError("VideoWriter failed to open for all attempted codecs/paths.")


def estimate_fps_from_timestamps(timestamps: list[float], fallback_fps: float) -> float:
    if len(timestamps) < 2:
        return float(fallback_fps)

    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0:
        return float(fallback_fps)

    est = (len(timestamps) - 1) / elapsed
    return max(1.0, est)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=str, help="Path to INT8 ONNX model (ssd_int8.onnx)")
    ap.add_argument("--camera", default=0, type=int, help="Webcam index")
    ap.add_argument("--width", default=1280, type=int)
    ap.add_argument("--height", default=720, type=int)
    ap.add_argument("--fps", default=30, type=int, help="Requested camera FPS")
    ap.add_argument("--record-fps", default=0.0, type=float,
                    help="Saved video FPS. Use 0 for auto-estimate from actual processed frame rate.")
    ap.add_argument("--record-init-frames", default=30, type=int,
                    help="Number of processed frames to observe before auto-selecting saved video FPS.")
    ap.add_argument("--score-thresh", default=0.20, type=float)
    ap.add_argument("--nms-thresh", default=0.50, type=float)
    ap.add_argument("--max-per-img", default=100, type=int)
    ap.add_argument("--class-agnostic", action="store_true")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--show-fps", action="store_true")
    ap.add_argument("--save-video", action="store_true", help="Save annotated output to a video file")
    ap.add_argument("--out-video", default="ssd_int8_demo.mp4", type=str, help="Output video path")
    ap.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "any", "dshow", "msmf", "v4l2", "gstreamer"],
        help="VideoCapture backend. Use 'auto' for OS-specific fallback.",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="Optional device path (Linux), e.g. /dev/video2. If set, overrides --camera.",
    )
    args = ap.parse_args()

    class_to_idx = {"biker": 0, "car": 1, "pedestrian": 2, "trafficLight": 3, "truck": 4}

    predictor = SSDInt8ONNXPredictor(
        onnx_model_path=args.model,
        class_to_idx=class_to_idx,
        providers=["CPUExecutionProvider"],
        preprocess_cfg=PreprocessConfig(input_color="bgr"),
    )

    source = args.device if args.device is not None else args.camera
    cap, backend_used = open_camera(source, args.backend)
    print(f"[info] opened camera source={source!r} using backend={backend_used}")
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    cap.set(cv2.CAP_PROP_FPS, float(args.fps))

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    ok, frame_bgr = cap.read()
    if not ok:
        raise RuntimeError("Failed to read initial frame.")
    _ = predictor(frame_bgr)

    writer = None
    out_path = args.out_video
    buffered_frames: list[np.ndarray] = []
    buffered_timestamps: list[float] = []
    record_fps = float(args.record_fps) if args.record_fps > 0 else None

    fps_smoothed = 0.0
    last_print = time.perf_counter()

    try:
        while True:
            loop_t0 = time.perf_counter()

            ok, frame_bgr = cap.read()
            if not ok:
                break

            pred = predictor(frame_bgr)
            vis = draw_predictions_bgr(frame_bgr, pred, show_labels=not args.no_labels)

            dt = time.perf_counter() - loop_t0
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

            if args.save_video:
                if writer is None:
                    buffered_frames.append(vis.copy())
                    buffered_timestamps.append(time.perf_counter())

                    ready_to_open = record_fps is not None or len(buffered_frames) >= max(2, args.record_init_frames)
                    if ready_to_open:
                        if record_fps is None:
                            record_fps = estimate_fps_from_timestamps(
                                buffered_timestamps,
                                fallback_fps=max(1.0, fps_smoothed, float(args.fps)),
                            )

                        H0, W0 = buffered_frames[0].shape[:2]
                        writer, out_path = open_video_writer(out_path, record_fps, (W0, H0))
                        print(f"[info] saving video at {record_fps:.2f} FPS -> {out_path}")

                        for fr in buffered_frames:
                            writer.write(fr)

                        buffered_frames.clear()
                        buffered_timestamps.clear()
                else:
                    writer.write(vis)

            cv2.imshow("SSD INT8 ONNX Runtime", vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            now = time.perf_counter()
            if now - last_print > 5.0:
                print(f"[info] smoothed FPS ~ {fps_smoothed:.1f} | dets={len(pred['labels'])}")
                last_print = now

    finally:
        if args.save_video and writer is None and buffered_frames:
            if record_fps is None:
                record_fps = estimate_fps_from_timestamps(
                    buffered_timestamps,
                    fallback_fps=max(1.0, fps_smoothed, float(args.fps)),
                )
            H0, W0 = buffered_frames[0].shape[:2]
            writer, out_path = open_video_writer(out_path, record_fps, (W0, H0))
            print(f"[info] saving short video at {record_fps:.2f} FPS -> {out_path}")
            for fr in buffered_frames:
                writer.write(fr)

        if writer is not None:
            writer.release()
            print(f"Saved video: {out_path}")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
