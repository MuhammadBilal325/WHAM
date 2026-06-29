"""Visualize SMPL results from a saved output .npy file.

Loads a .npy file containing SMPL results (vertices, joints_3d, betas, etc.)
and renders them as a video.

Usage:
    python tests/visualize_output.py <path_to_output_npy> [--output <output_path>] [--fps 30] [--image_size 512] [--mode mesh]
"""

import os
import sys
import argparse
import numpy as np

# Add project root to path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PROJECT_ROOT)

from utils.common.visualizer import visualize_smpl_results


def main():
    parser = argparse.ArgumentParser(description='Visualize SMPL results from output npy file')
    parser.add_argument('npy_path', type=str, help='Path to the output .npy file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output video path (default: <input_name>_vis.mp4)')
    parser.add_argument('--device', '-d', type=str, default='cpu', choices=['cuda', 'cpu'],
                        help='Device to use for loading body model (default: cpu)')
    parser.add_argument('--fps', type=int, default=30, help='Output video FPS (default: 30)')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Output frame size in pixels (default: 512)')
    parser.add_argument('--mode', '-m', type=str, default='mesh',
                        choices=['mesh', 'wireframe', 'skeleton', 'mesh+skeleton'],
                        help='Visualization mode (default: mesh)')
    args = parser.parse_args()

    if not os.path.exists(args.npy_path):
        print(f"Error: File not found: {args.npy_path}")
        sys.exit(1)

    device = args.device

    print(f"Loading data from: {args.npy_path}")
    data = np.load(args.npy_path, allow_pickle=True)

    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()
    elif isinstance(data, np.lib.npyio.NpzFile):
        data = dict(data)

    print(f"Available keys: {list(data.keys())}")
    for key in data.keys():
        print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")

    if args.output is None:
        base = os.path.splitext(args.npy_path)[0]
        args.output = base + '_vis.mp4'

    visualize_smpl_results(data, args.output, device=device, fps=args.fps,
                           image_size=args.image_size, mode=args.mode)


if __name__ == '__main__':
    main()
