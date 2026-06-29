"""Realtime SMPL lifting server.

Receives datagram inputs (2D COCO17 keypoints) over UDP from a given IP:port,
lifts them to SMPL parameters using WHAM, and sends compact output datagrams
(SMPL thetas + predicted camera root position) to a given IP:port.

The output datagram is a compact binary format:
  [root_rot6d (6 floats)][body_pose_rot6d (138 floats)][betas (10 floats)]
  Total: 154 floats = 616 bytes per frame

Usage:
    python main.py [--input_ip 0.0.0.0] [--input_port 5005] [--output_ip 127.0.0.1] [--output_port 5006]
                   [--device cuda] [--window_size 16] [--verbose]
"""

import os
import sys
import argparse
import socket
import struct
import time
import signal
import threading

import numpy as np

# Add src/ to path so we can import utils.common directly
_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _PROJECT_ROOT)

from utils.common.datagram import (
    parse_input_datagram,
    serialize_output_datagram,
    OUTPUT_BYTES,
)
from utils.common.lifter import CameraSpaceLifter


class RealtimeSmplServer:
    """Realtime UDP server for SMPL lifting.

    Listens for incoming datagrams on input_ip:input_port,
    lifts them through WHAM, and sends compact SMPL output
    to output_ip:output_port.
    """

    def __init__(self, input_ip='0.0.0.0', input_port=5005,
                 output_ip='127.0.0.1', output_port=5006,
                 device='cuda', window_size=16, verbose=False):
        self.input_ip = input_ip
        self.input_port = input_port
        self.output_ip = output_ip
        self.output_port = output_port
        self.verbose = verbose
        self.running = False

        # Initialize lifter
        print(f"Initializing WHAM lifter on device: {device}")
        self.lifter = CameraSpaceLifter(device=device, window_size=window_size)

        # Socket setup
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # 1MB receive buffer
        self.recv_sock.bind((self.input_ip, self.input_port))

        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Stream state
        self.stream_started = False
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0.0

        print(f"Server configured:")
        print(f"  Input:  {self.input_ip}:{self.input_port}")
        print(f"  Output: {self.output_ip}:{self.output_port}")
        print(f"  Output datagram size: {OUTPUT_BYTES} bytes")

    def start(self):
        """Start the server loop."""
        self.running = True
        print(f"\nServer listening on {self.input_ip}:{self.input_port}...")
        print(f"Press Ctrl+C to stop.\n")

        # Set socket to non-blocking with timeout for graceful shutdown
        self.recv_sock.settimeout(0.5)

        while self.running:
            try:
                data, addr = self.recv_sock.recvfrom(4096)
                self._handle_datagram(data, addr)
            except socket.timeout:
                continue
            except OSError as e:
                if self.running:
                    print(f"Socket error: {e}")
                break

    def _handle_datagram(self, data, addr):
        """Process a single incoming datagram."""
        try:
            # Parse input datagram
            parsed = parse_input_datagram(data)

            # Initialize stream on first frame
            if not self.stream_started:
                image_width = int(parsed['image_width'])
                image_height = int(parsed['image_height'])
                self.lifter.start_stream(image_width, image_height)
                self.stream_started = True
                if self.verbose:
                    print(f"Stream initialized: {image_width}x{image_height}")

            # Extract keypoints and confidence
            kp2d = parsed.get('pose_2d')
            if kp2d is None:
                self.dropped_frames += 1
                return

            confidence = parsed.get('confidence_2d', np.ones(17))

            # Run lifting
            result = self.lifter.push_frame(kp2d, confidence)

            if result is not None:
                # Serialize and send output
                output_bytes = serialize_output_datagram(
                    root_rot6d=result['root_rot6d'],
                    body_pose_rot6d=result['body_pose_rot6d'],
                    betas=result['betas'].flatten(),
                )
                self.send_sock.sendto(output_bytes, (self.output_ip, self.output_port))

                # Update stats
                self.frame_count += 1
                self.fps_counter += 1

                now = time.time()
                elapsed = now - self.last_fps_time
                if elapsed >= 1.0:
                    self.current_fps = self.fps_counter / elapsed
                    self.fps_counter = 0
                    self.last_fps_time = now
                    if self.verbose:
                        print(f"  FPS: {self.current_fps:.1f} | Frames: {self.frame_count} | Dropped: {self.dropped_frames}")

        except Exception as e:
            self.dropped_frames += 1
            if self.verbose:
                print(f"Error processing datagram: {e}")

    def stop(self):
        """Stop the server."""
        self.running = False
        self.recv_sock.close()
        self.send_sock.close()
        print(f"\nServer stopped. Processed {self.frame_count} frames, dropped {self.dropped_frames}.")


def main():
    parser = argparse.ArgumentParser(description='Realtime SMPL lifting server')
    parser.add_argument('--input_ip', type=str, default='0.0.0.0',
                        help='IP address to listen on (default: 0.0.0.0)')
    parser.add_argument('--input_port', type=int, default=5005,
                        help='Port to listen on (default: 5005)')
    parser.add_argument('--output_ip', type=str, default='127.0.0.1',
                        help='IP address to send output to (default: 127.0.0.1)')
    parser.add_argument('--output_port', type=int, default=5006,
                        help='Port to send output to (default: 5006)')
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device for WHAM inference (default: cuda)')
    parser.add_argument('--window_size', '-w', type=int, default=16,
                        help='Sliding window size for temporal smoothing (default: 16)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output (FPS stats, errors)')
    args = parser.parse_args()

    # Check CUDA availability
    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            args.device = 'cpu'

    server = RealtimeSmplServer(
        input_ip=args.input_ip,
        input_port=args.input_port,
        output_ip=args.output_ip,
        output_port=args.output_port,
        device=args.device,
        window_size=args.window_size,
        verbose=args.verbose,
    )

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    server.start()


if __name__ == '__main__':
    main()
