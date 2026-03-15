import argparse
import os
import os.path as osp
import queue
import threading
import time
from collections import defaultdict
from collections import deque

import cv2
import joblib
import numpy as np
import torch
from loguru import logger

from configs.config import get_cfg_defaults
from lib.data.datasets import CustomDataset
from lib.models import build_body_model, build_network
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor

"""
Realtime WHAM from UDP webcam stream.

Example sender (run outside this script):
ffmpeg -f v4l2 -framerate 30 -video_size 1280x720 -i /dev/video0 \
    -pix_fmt yuv420p -f mpegts udp://127.0.0.1:23000?pkt_size=1316

Example receiver:
python webcam.py --stream_url "udp://127.0.0.1:23000?fifo_size=1000000&overrun_nonfatal=1"
"""


class UdpFrameGrabber:
    """Read frames from a UDP stream in a background thread with bounded buffering."""

    def __init__(self, stream_url, max_queue_size=256, input_mode="udp", webcam_index=0):
        self.stream_url = stream_url
        self.max_queue_size = max_queue_size
        self.input_mode = input_mode
        self.webcam_index = webcam_index
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.thread = None
        self.cap = None
        self.frames_received = 0
        self.frames_dropped = 0
        self.frames_skipped_latest = 0
        self.failed_reads = 0
        self.reopen_fail_read_threshold = 300

    def _open_capture(self):
        source = self.stream_url if self.input_mode == "udp" else int(self.webcam_index)
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if cap.isOpened():
            return cap

        # Fallback to default backend if FFmpeg-specific open fails.
        cap.release()
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return cap

        cap.release()
        return None

    def _reopen_capture_blocking(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        while not self.stop_event.is_set():
            cap = self._open_capture()
            if cap is not None:
                self.cap = cap
                self.failed_reads = 0
                logger.info(f"Input opened (mode={self.input_mode})")
                return

            logger.warning(
                f"Input not available yet (mode={self.input_mode}, source={self.stream_url if self.input_mode == 'udp' else self.webcam_index}). Retrying in 2s..."
            )
            time.sleep(2.0)

    def start(self):
        # OpenCV+FFmpeg may time out if sender is not yet publishing. Keep retrying.
        self._reopen_capture_blocking()

        if self.cap is None:
            raise RuntimeError("Stream grabber stopped before stream opened")

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self.failed_reads += 1
                if self.failed_reads >= self.reopen_fail_read_threshold:
                    logger.warning("Too many failed reads, reopening UDP stream...")
                    self._reopen_capture_blocking()
                time.sleep(0.005)
                continue

            self.failed_reads = 0

            self.frames_received += 1
            item = (self.frames_received - 1, frame, time.time())

            if self.queue.full():
                try:
                    self.queue.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass

            try:
                self.queue.put_nowait(item)
            except queue.Full:
                self.frames_dropped += 1

    def _drain_queue_nowait(self, on_item=None):
        drained = []
        while True:
            try:
                item = self.queue.get_nowait()
                drained.append(item)
                if on_item is not None:
                    on_item(item)
            except queue.Empty:
                break
        return drained

    def pop_chunk(self, chunk_size, timeout_s=5.0, on_item=None, latest_only=False):
        chunk = []
        start = time.time()
        while len(chunk) < chunk_size:
            remaining = max(0.0, timeout_s - (time.time() - start))
            if remaining <= 0:
                break
            try:
                item = self.queue.get(timeout=remaining)
                chunk.append(item)
                if on_item is not None:
                    on_item(item)
            except queue.Empty:
                break

        # Low-latency mode: keep only the newest chunk by draining stale backlog.
        if latest_only and len(chunk) > 0:
            chunk.extend(self._drain_queue_nowait(on_item=on_item))
            if len(chunk) > chunk_size:
                dropped = len(chunk) - chunk_size
                self.frames_skipped_latest += dropped
                chunk = chunk[-chunk_size:]

        return chunk

    def pop_latest(self, timeout_s=0.5, on_item=None):
        """Pop the newest available frame, dropping older queued frames."""
        try:
            item = self.queue.get(timeout=timeout_s)
            if on_item is not None:
                on_item(item)
        except queue.Empty:
            return None

        drained = self._drain_queue_nowait(on_item=on_item)
        if len(drained) > 0:
            self.frames_skipped_latest += len(drained)
            item = drained[-1]
        return item

    def wait_for_first_frame(self, poll_interval_s=0.5):
        """Block indefinitely until at least one decoded frame is available."""
        while not self.stop_event.is_set():
            try:
                return self.queue.get(timeout=poll_interval_s)
            except queue.Empty:
                continue
        return None

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


class RealtimeWham:
    def __init__(self, cfg, vitpose_variant='h', vitpose_ckpt=None, min_track_frames=8, min_valid_joints=6):
        self.cfg = cfg
        smpl_batch = self.cfg.TRAIN.BATCH_SIZE * self.cfg.DATASET.SEQLEN
        self.network = build_network(self.cfg, build_body_model(self.cfg.DEVICE, smpl_batch))
        self.network.eval()
        self.detector = DetectionModel(
            self.cfg.DEVICE.lower(),
            vitpose_variant=vitpose_variant,
            vitpose_ckpt=vitpose_ckpt,
            minimum_frames=min_track_frames,
            minimum_joints=min_valid_joints,
        )
        self.extractor = FeatureExtractor(self.cfg.DEVICE.lower(), self.cfg.FLIP_EVAL)

    @torch.no_grad()
    def run_chunk(self, frames, fps):
        if len(frames) == 0:
            return {}, {}, None

        height, width = frames[0].shape[:2]

        # Reset tracking state while keeping the detection model in memory.
        self.detector.initialize_tracking()
        length = len(frames)
        for frame in frames:
            self.detector.track(frame, fps=fps, length=length)

        # No person detected in this chunk; skip model pipeline safely.
        if len(self.detector.tracking_results["keypoints"]) == 0:
            return {}, {}, {
                "width": width,
                "height": height,
                "n_frames": length,
                "n_subjects": 0,
            }

        tracking_results = self.detector.process(fps)
        if len(tracking_results) == 0:
            return {}, {}, {
                "width": width,
                "height": height,
                "n_frames": length,
                "n_subjects": 0,
            }

        # Stationary camera assumption: camera orientation is fixed.
        slam_results = np.zeros((length, 7), dtype=np.float32)
        slam_results[:, 3] = 1.0

        tracking_results = self.extractor.run(frames, tracking_results)

        dataset = CustomDataset(self.cfg, tracking_results, slam_results, width, height, fps)
        results = defaultdict(dict)

        for subj_idx in range(len(dataset)):
            batch = dataset.load_data(subj_idx)
            if batch is None:
                continue

            _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
            pred = self.network(
                x,
                inits,
                features,
                mask=mask,
                init_root=init_root,
                cam_angvel=cam_angvel,
                return_y_up=True,
                **kwargs,
            )

            results[_id]["pose_root_cam"] = pred["poses_root_cam"].cpu().squeeze(0).numpy()
            results[_id]["pose_body"] = pred["poses_body"].cpu().squeeze(0).numpy()
            results[_id]["betas"] = pred["betas"].cpu().squeeze(0).numpy()
            results[_id]["trans_world"] = pred["trans_world"].cpu().squeeze(0).numpy()
            results[_id]["verts_cam"] = (
                pred["verts_cam"] + pred["trans_cam"].unsqueeze(1)
            ).cpu().squeeze(0).numpy()
            results[_id]["frame_id"] = frame_id

        meta = {
            "width": width,
            "height": height,
            "n_frames": length,
            "n_subjects": len(results),
        }
        return results, tracking_results, meta


class AsyncLatestProcessor:
    def __init__(self, engine, cfg, preview_enabled, save_results, output_dir):
        self.engine = engine
        self.cfg = cfg
        self.preview_enabled = preview_enabled
        self.save_results = save_results
        self.output_dir = output_dir

        self._job_lock = threading.Lock()
        self._latest_job = None
        self._job_event = threading.Event()
        self._stop_event = threading.Event()

        self._out_lock = threading.Lock()
        self._latest_overlay = None
        self._latest_text = None
        self._latest_chunk_id = -1

        self._renderer = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, chunk_id, frames, frame_indices, stream_ts, fps):
        job = {
            "chunk_id": chunk_id,
            "frames": [f.copy() for f in frames],
            "frame_indices": list(frame_indices),
            "stream_ts": list(stream_ts),
            "fps": fps,
        }
        with self._job_lock:
            self._latest_job = job
            self._job_event.set()

    def get_latest_overlay(self):
        with self._out_lock:
            if self._latest_overlay is None:
                return None, None, -1
            return self._latest_overlay.copy(), self._latest_text, self._latest_chunk_id

    def stop(self):
        self._stop_event.set()
        self._job_event.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop_event.is_set():
            self._job_event.wait(timeout=0.1)
            if self._stop_event.is_set():
                break
            if not self._job_event.is_set():
                continue

            with self._job_lock:
                job = self._latest_job
                self._latest_job = None
                self._job_event.clear()

            if job is None:
                continue

            chunk_id = job["chunk_id"]
            frames = job["frames"]
            frame_indices = job["frame_indices"]
            stream_ts = job["stream_ts"]
            fps = job["fps"]

            tic = time.time()
            results, tracking_results, meta = self.engine.run_chunk(frames, fps=fps)
            infer_time = time.time() - tic
            end_to_end_latency = time.time() - stream_ts[-1]
            processed_fps = (meta["n_frames"] / infer_time) if infer_time > 0 else 0.0

            logger.info(
                "chunk={} frames=[{}..{}] subjects={} infer={:.2f}s chunk_fps={:.2f} stream_latency={:.2f}s"
                .format(
                    chunk_id,
                    frame_indices[0],
                    frame_indices[-1],
                    meta["n_subjects"],
                    infer_time,
                    processed_fps,
                    end_to_end_latency,
                )
            )

            overlay_bgr = None
            overlay_text = None
            if self.preview_enabled:
                if self._renderer is None:
                    from lib.vis.renderer import Renderer

                    focal_length = (meta["width"] ** 2 + meta["height"] ** 2) ** 0.5
                    self._renderer = Renderer(
                        meta["width"],
                        meta["height"],
                        focal_length,
                        self.cfg.DEVICE,
                        self.engine.network.smpl.faces,
                    )

                rgb = frames[-1][..., ::-1].copy()
                newest_local_idx = len(frames) - 1
                for _, val in results.items():
                    frame_ids = val.get("frame_id", [])
                    verts_cam = val.get("verts_cam", [])
                    hits = np.where(frame_ids == newest_local_idx)[0]
                    if len(hits) == 0:
                        continue
                    i = hits[-1]
                    if i >= len(verts_cam):
                        continue
                    mesh_verts = torch.from_numpy(verts_cam[i]).float().to(self.cfg.DEVICE)
                    rgb = self._renderer.render_mesh(mesh_verts, rgb)

                overlay_bgr = np.clip(rgb, 0, 255).astype(np.uint8)[..., ::-1]
                overlay_text = (
                    f"mesh | chunk {chunk_id} | subj {meta['n_subjects']} | "
                    f"infer {infer_time:.2f}s | lag {end_to_end_latency:.2f}s"
                )

            if self.save_results:
                out_file = osp.join(self.output_dir, f"wham_chunk_{chunk_id:06d}.pth")
                payload = {
                    "meta": {
                        **meta,
                        "chunk_id": chunk_id,
                        "frame_start": frame_indices[0],
                        "frame_end": frame_indices[-1],
                        "infer_time_s": infer_time,
                        "processed_fps": processed_fps,
                        "stream_latency_s": end_to_end_latency,
                    },
                    "results": results,
                    "tracking_results": tracking_results,
                }
                joblib.dump(payload, out_file)

            with self._out_lock:
                self._latest_overlay = overlay_bgr
                self._latest_text = overlay_text
                self._latest_chunk_id = chunk_id


def build_cfg():
    cfg = get_cfg_defaults()
    cfg.merge_from_file("configs/yamls/demo.yaml")
    # Flip evaluation doubles inference cost and increases latency.
    cfg.FLIP_EVAL = False
    if cfg.DEVICE.lower().startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested in config but not available. Falling back to CPU.")
        cfg.DEVICE = "cpu"
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime WHAM from UDP webcam stream")
    parser.add_argument(
        "--input_mode",
        type=str,
        choices=["udp", "webcam"],
        default="udp",
        help="Input source mode: udp stream URL or webcam device index",
    )
    parser.add_argument(
        "--webcam_index",
        type=int,
        default=0,
        help="OpenCV webcam device index when --input_mode webcam",
    )
    parser.add_argument(
        "--stream_url",
        type=str,
        default="udp://127.0.0.1:23000?fifo_size=1000000&overrun_nonfatal=1",
        help="OpenCV-compatible UDP input URL",
    )
    parser.add_argument(
        "--vitpose_variant",
        type=str,
        choices=["h", "b", "l", "s"],
        default="h",
        help="ViTPose model variant: h=huge, b=base, l=large, s=small",
    )
    parser.add_argument(
        "--vitpose_ckpt",
        type=str,
        default=None,
        help="Optional path to ViTPose checkpoint. If omitted, defaults to checkpoints/vitpose-<variant>-multi-coco.pth",
    )
    parser.add_argument(
        "--min_track_frames",
        type=int,
        default=8,
        help="Minimum per-subject tracked frames to keep after detector postprocess",
    )
    parser.add_argument(
        "--min_valid_joints",
        type=int,
        default=6,
        help="Minimum visible joints required to accept a pose detection",
    )
    parser.add_argument("--output_dir", type=str, default="output/webcam", help="Output folder")
    parser.add_argument("--chunk_size", type=int, default=64, help="Frames per WHAM inference chunk")
    parser.add_argument("--fps", type=float, default=30.0, help="Fallback FPS if stream FPS is unavailable")
    parser.add_argument(
        "--max_queue_frames",
        type=int,
        default=256,
        help="Maximum buffered frames before dropping old frames",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save per-chunk WHAM outputs as joblib files",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=0,
        help="Stop after this many chunks. 0 means run forever.",
    )
    parser.add_argument(
        "--preview",
        dest="preview",
        action="store_true",
        help="Show live preview with mesh overlay",
    )
    parser.add_argument(
        "--no_preview",
        dest="preview",
        action="store_false",
        help="Disable preview window",
    )
    parser.add_argument(
        "--frames_only",
        action="store_true",
        help="Do not load any models; only display incoming frames",
    )
    parser.add_argument(
        "--latest_only",
        action="store_true",
        help="Process only the newest chunk (drop stale buffered frames for lower latency)",
    )
    parser.add_argument(
        "--process_oldest",
        dest="latest_only",
        action="store_false",
        help="Process chunks in FIFO order (can increase lag)",
    )
    parser.add_argument(
        "--inference_stride",
        type=int,
        default=4,
        help="Submit one inference job every N captured frames",
    )
    parser.set_defaults(latest_only=True)
    parser.set_defaults(preview=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = None
    engine = None
    if not args.frames_only:
        cfg = build_cfg()

        logger.info(f"Using device: {cfg.DEVICE}")
        if torch.cuda.is_available() and cfg.DEVICE.lower().startswith("cuda"):
            logger.info(f"GPU name -> {torch.cuda.get_device_name()}")

        engine = RealtimeWham(
            cfg,
            vitpose_variant=args.vitpose_variant,
            vitpose_ckpt=args.vitpose_ckpt,
            min_track_frames=args.min_track_frames,
            min_valid_joints=args.min_valid_joints,
        )
    else:
        logger.info("Running in frames-only mode: model loading and predictions are disabled")
    grabber = UdpFrameGrabber(
        args.stream_url,
        max_queue_size=args.max_queue_frames,
        input_mode=args.input_mode,
        webcam_index=args.webcam_index,
    )

    logger.info("Starting UDP frame grabber")
    grabber.start()

    logger.info("Waiting indefinitely for sender / first valid frame...")
    first_item = grabber.wait_for_first_frame()
    if first_item is None:
        raise RuntimeError("Stream grabber stopped before receiving any frames")
    try:
        grabber.queue.put_nowait(first_item)
    except queue.Full:
        # If the queue is full, dropping one warmup frame is acceptable.
        pass
    logger.info("First frame received. Starting WHAM processing loop.")

    renderer = None
    preview_enabled = args.preview
    preview_window = "WHAM Realtime Preview"
    if preview_enabled and (not args.frames_only):
        try:
            from lib.vis.renderer import Renderer
        except Exception as exc:
            logger.warning(f"Preview disabled: failed to import renderer ({exc})")
            preview_enabled = False

    chunk_id = 0
    t0 = time.time()
    stop_requested = False
    rolling = deque(maxlen=args.chunk_size)
    capture_counter = 0
    last_rendered_chunk_id = -1
    last_qskip_total = 0

    processor = None
    if (not args.frames_only) and engine is not None:
        processor = AsyncLatestProcessor(
            engine=engine,
            cfg=cfg,
            preview_enabled=preview_enabled,
            save_results=args.save_results,
            output_dir=args.output_dir,
        )

    def show_preview_frame(frame_bgr, overlay_text=None):
        nonlocal stop_requested
        if not args.preview:
            return

        disp = frame_bgr.copy()
        if overlay_text is not None:
            cv2.putText(
                disp,
                overlay_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow(preview_window, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            logger.info("Preview requested stop (q pressed)")
            stop_requested = True

    def on_chunk_item(item):
        # Show live raw frames during chunk collection for smoother preview.
        _, frame, _ = item
        if args.frames_only:
            text = "frames-only | collecting"
        else:
            text = "collecting frames"
        show_preview_frame(frame, text)

    try:
        while True:
            if stop_requested:
                break
            if args.max_chunks > 0 and chunk_id >= args.max_chunks:
                break

            item = grabber.pop_latest(timeout_s=0.5)
            if item is None:
                continue

            frame_idx, frame, frame_ts = item
            rolling.append((frame_idx, frame, frame_ts))
            capture_counter += 1

            if args.frames_only:
                if args.preview:
                    show_preview_frame(
                        frame,
                        f"frames-only | frame {frame_idx} | qdrop {grabber.frames_dropped} | qskip {grabber.frames_skipped_latest}",
                    )
                continue

            if len(rolling) >= args.chunk_size and (capture_counter % max(1, args.inference_stride) == 0):
                window = list(rolling)
                if args.latest_only and len(window) > args.chunk_size:
                    window = window[-args.chunk_size:]

                frame_indices = [x[0] for x in window]
                frames = [x[1] for x in window]
                stream_ts = [x[2] for x in window]

                processor.submit(
                    chunk_id=chunk_id,
                    frames=frames,
                    frame_indices=frame_indices,
                    stream_ts=stream_ts,
                    fps=args.fps,
                )
                chunk_id += 1

            if args.preview:
                overlay, overlay_text, overlay_chunk_id = processor.get_latest_overlay()
                if overlay is not None and overlay_chunk_id != last_rendered_chunk_id:
                    show_preview_frame(overlay, overlay_text)
                    last_rendered_chunk_id = overlay_chunk_id
                else:
                    qskip_delta = grabber.frames_skipped_latest - last_qskip_total
                    last_qskip_total = grabber.frames_skipped_latest
                    show_preview_frame(
                        frame,
                        f"live | frame {frame_idx} | rolling {len(rolling)}/{args.chunk_size} | qdrop {grabber.frames_dropped} | qskip+{qskip_delta} (total {grabber.frames_skipped_latest})",
                    )

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if processor is not None:
            processor.stop()
        grabber.stop()
        if args.preview:
            cv2.destroyAllWindows()
        elapsed = time.time() - t0
        logger.info(f"Finished. chunks={chunk_id}, elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    main()
