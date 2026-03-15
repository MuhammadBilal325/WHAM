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
        if self.input_mode == "webcam":
            cap = cv2.VideoCapture(int(self.webcam_index))
            if cap.isOpened():
                return cap
            cap.release()
            return None

        cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            return cap

        # Fallback to default backend if FFmpeg-specific open fails.
        cap.release()
        cap = cv2.VideoCapture(self.stream_url)
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
                source_name = (
                    f"webcam:{self.webcam_index}"
                    if self.input_mode == "webcam"
                    else self.stream_url
                )
                logger.info(f"Video stream opened ({source_name})")
                return

            source_name = (
                f"webcam:{self.webcam_index}"
                if self.input_mode == "webcam"
                else self.stream_url
            )
            logger.warning(
                f"Stream not available yet: {source_name}. Retrying in 2s..."
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
    def __init__(self, cfg):
        self.cfg = cfg
        smpl_batch = self.cfg.TRAIN.BATCH_SIZE * self.cfg.DATASET.SEQLEN
        self.network = build_network(self.cfg, build_body_model(self.cfg.DEVICE, smpl_batch))
        self.network.eval()
        self.detector = DetectionModel(self.cfg.DEVICE.lower())
        self.extractor = FeatureExtractor(self.cfg.DEVICE.lower(), self.cfg.FLIP_EVAL)

    @torch.no_grad()
    def run_chunk(self, frames, chunk_video_path, fps):
        if len(frames) == 0:
            return {}, {}, None

        height, width = frames[0].shape[:2]

        writer = cv2.VideoWriter(
            chunk_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create temporary video: {chunk_video_path}")

        for frame in frames:
            writer.write(frame)
        writer.release()

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

        tracking_results = self.extractor.run(chunk_video_path, tracking_results)

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
        default="udp",
        choices=["udp", "webcam"],
        help="Input source type: udp stream or direct webcam device",
    )
    parser.add_argument(
        "--stream_url",
        type=str,
        default="udp://127.0.0.1:23000?fifo_size=1000000&overrun_nonfatal=1",
        help="OpenCV-compatible UDP input URL",
    )
    parser.add_argument(
        "--webcam_index",
        type=int,
        default=0,
        help="Webcam device index when --input_mode webcam",
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
        "--queue_timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for new frames in streaming loop",
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
        "--keep_temp_videos",
        action="store_true",
        help="Keep temporary chunk videos for debugging",
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
        help="Process buffered windows in FIFO order (can increase lag)",
    )
    parser.add_argument(
        "--infer_stride",
        type=int,
        default=2,
        help="Submit one inference task every N incoming frames",
    )
    parser.set_defaults(latest_only=True)
    parser.set_defaults(preview=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = osp.join(args.output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    cfg = None
    engine = None
    if not args.frames_only:
        cfg = build_cfg()

        logger.info(f"Using device: {cfg.DEVICE}")
        if torch.cuda.is_available() and cfg.DEVICE.lower().startswith("cuda"):
            logger.info(f"GPU name -> {torch.cuda.get_device_name()}")

        engine = RealtimeWham(cfg)
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
    logger.info("First frame received. Starting streaming loop.")

    preview_window_live = "WHAM Live Input"
    preview_window_mesh = "WHAM Mesh Latest"

    preview_enabled = args.preview
    if preview_enabled and (not args.frames_only):
        try:
            from lib.vis.renderer import Renderer
        except Exception as exc:
            logger.warning(f"Mesh preview disabled: failed to import renderer ({exc})")
            preview_enabled = False
            Renderer = None
    else:
        Renderer = None

    submit_chunk_id = 0
    completed_chunks = 0
    frame_counter = 0
    rolling_buffer = deque(maxlen=max(2, args.chunk_size))
    rolling_buffer.append(first_item)
    t0 = time.time()
    stop_requested = False

    latest_mesh_preview = None
    latest_mesh_stats = None

    infer_queue = queue.Queue(maxsize=1)
    infer_results_queue = queue.Queue(maxsize=1)
    worker_stop_event = threading.Event()

    def put_latest(q_obj, value):
        try:
            q_obj.put_nowait(value)
            return
        except queue.Full:
            pass

        try:
            q_obj.get_nowait()
        except queue.Empty:
            pass

        try:
            q_obj.put_nowait(value)
        except queue.Full:
            pass

    def inference_worker():
        renderer = None
        while not worker_stop_event.is_set():
            try:
                task = infer_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if task is None:
                break

            task_chunk_id, chunk_items = task
            frame_indices = [x[0] for x in chunk_items]
            frames = [x[1] for x in chunk_items]
            stream_ts = [x[2] for x in chunk_items]
            chunk_video_path = osp.join(temp_dir, f"chunk_{task_chunk_id:06d}.mp4")

            tic = time.time()
            results, tracking_results, meta = engine.run_chunk(frames, chunk_video_path, fps=args.fps)
            infer_time = time.time() - tic
            end_to_end_latency = time.time() - stream_ts[-1]
            processed_fps = (meta["n_frames"] / infer_time) if infer_time > 0 else 0.0

            preview_bgr = None
            if preview_enabled and Renderer is not None:
                try:
                    if renderer is None:
                        focal_length = (meta["width"] ** 2 + meta["height"] ** 2) ** 0.5
                        renderer = Renderer(
                            meta["width"],
                            meta["height"],
                            focal_length,
                            cfg.DEVICE,
                            engine.network.smpl.faces,
                        )

                    # Render only the newest frame in the window to reduce preview lag.
                    local_last_idx = len(frames) - 1
                    rgb = frames[local_last_idx][..., ::-1].copy()
                    for _, val in results.items():
                        frame_ids = np.asarray(val.get("frame_id", []))
                        verts_cam = val.get("verts_cam", [])
                        if frame_ids.size == 0:
                            continue
                        match = np.where(frame_ids == local_last_idx)[0]
                        if len(match) == 0:
                            continue
                        mesh_idx = int(match[-1])
                        if mesh_idx >= len(verts_cam):
                            continue
                        mesh_verts = torch.from_numpy(verts_cam[mesh_idx]).float().to(cfg.DEVICE)
                        rgb = renderer.render_mesh(mesh_verts, rgb)

                    preview_bgr = np.clip(rgb, 0, 255).astype(np.uint8)[..., ::-1]
                except Exception as exc:
                    logger.warning(f"Mesh render failed for chunk {task_chunk_id}: {exc}")

            payload = {
                "chunk_id": task_chunk_id,
                "frame_start": frame_indices[0],
                "frame_end": frame_indices[-1],
                "results": results,
                "tracking_results": tracking_results,
                "meta": meta,
                "infer_time": infer_time,
                "processed_fps": processed_fps,
                "stream_latency": end_to_end_latency,
                "preview_bgr": preview_bgr,
            }
            put_latest(infer_results_queue, payload)

            if (not args.keep_temp_videos) and osp.exists(chunk_video_path):
                os.remove(chunk_video_path)

    worker_thread = None
    if not args.frames_only:
        worker_thread = threading.Thread(target=inference_worker, daemon=True)
        worker_thread.start()

    def show_preview_frame(frame_bgr, window_name, overlay_text=None):
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
        cv2.imshow(window_name, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            logger.info("Preview requested stop (q pressed)")
            stop_requested = True

    try:
        while True:
            if stop_requested:
                break
            if args.max_chunks > 0 and completed_chunks >= args.max_chunks:
                break

            try:
                item = grabber.queue.get(timeout=args.queue_timeout)
            except queue.Empty:
                continue

            rolling_buffer.append(item)
            frame_counter += 1
            _, live_frame, _ = item

            # Consume newest worker result.
            if not args.frames_only:
                while True:
                    try:
                        worker_result = infer_results_queue.get_nowait()
                    except queue.Empty:
                        break

                    latest_mesh_preview = worker_result.get("preview_bgr")
                    latest_mesh_stats = worker_result
                    completed_chunks += 1

                    logger.info(
                        "chunk={} frames=[{}..{}] subjects={} infer={:.2f}s chunk_fps={:.2f} stream_latency={:.2f}s queue_drop={} latest_skip={}"
                        .format(
                            worker_result["chunk_id"],
                            worker_result["frame_start"],
                            worker_result["frame_end"],
                            worker_result["meta"]["n_subjects"],
                            worker_result["infer_time"],
                            worker_result["processed_fps"],
                            worker_result["stream_latency"],
                            grabber.frames_dropped,
                            grabber.frames_skipped_latest,
                        )
                    )

                    if args.save_results:
                        out_file = osp.join(args.output_dir, f"wham_chunk_{worker_result['chunk_id']:06d}.pth")
                        joblib.dump(
                            {
                                "meta": {
                                    **worker_result["meta"],
                                    "chunk_id": worker_result["chunk_id"],
                                    "frame_start": worker_result["frame_start"],
                                    "frame_end": worker_result["frame_end"],
                                    "infer_time_s": worker_result["infer_time"],
                                    "processed_fps": worker_result["processed_fps"],
                                    "stream_latency_s": worker_result["stream_latency"],
                                },
                                "results": worker_result["results"],
                                "tracking_results": worker_result["tracking_results"],
                            },
                            out_file,
                        )

            if args.frames_only:
                if args.preview:
                    show_preview_frame(
                        live_frame,
                        preview_window_live,
                        f"frames-only | frame {item[0]} | drop {grabber.frames_dropped}",
                    )

                logger.info(
                    "frames-only frame={} queue_drop={} latest_skip={}"
                    .format(
                        item[0],
                        grabber.frames_dropped,
                        grabber.frames_skipped_latest,
                    )
                )
                continue

            # Show live input continuously regardless of inference state.
            if args.preview:
                pending = infer_queue.qsize()
                show_preview_frame(
                    live_frame,
                    preview_window_live,
                    f"live frame {item[0]} | buffer {len(rolling_buffer)}/{rolling_buffer.maxlen} | pending {pending}",
                )

                if latest_mesh_preview is not None:
                    stat_text = "mesh latest"
                    if latest_mesh_stats is not None:
                        stat_text = (
                            f"mesh chunk {latest_mesh_stats['chunk_id']} | subj {latest_mesh_stats['meta']['n_subjects']}"
                            f" | infer {latest_mesh_stats['infer_time']:.2f}s"
                        )
                    show_preview_frame(latest_mesh_preview, preview_window_mesh, stat_text)

            # Submit rolling buffer snapshot to worker.
            stride = max(1, args.infer_stride)
            if len(rolling_buffer) >= rolling_buffer.maxlen and (frame_counter % stride == 0):
                snapshot = list(rolling_buffer)

                # Optionally submit newest slice if we have extra backlog.
                if args.latest_only and len(snapshot) > args.chunk_size:
                    snapshot = snapshot[-args.chunk_size:]

                put_latest(infer_queue, (submit_chunk_id, snapshot))
                submit_chunk_id += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        worker_stop_event.set()
        if not args.frames_only:
            put_latest(infer_queue, None)
            if worker_thread is not None:
                worker_thread.join(timeout=2.0)
        grabber.stop()
        if args.preview:
            cv2.destroyAllWindows()
        elapsed = time.time() - t0
        logger.info(
            f"Finished. submitted_chunks={submit_chunk_id}, completed_chunks={completed_chunks}, elapsed={elapsed:.2f}s"
        )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    main()
