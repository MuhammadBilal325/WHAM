"""Offline test: lift 2D keypoints from a datagram npy file to SMPL mesh.

Reads datagrams from an npy file, runs them through the WHAM lifter,
and saves the SMPL results to a .npy file.

Usage:
    python tests/file_input.py <path_to_npy_file> [--output <output_path>] [--window_size 16] [--stride 1] [--device cuda]
"""

import os
import sys
import argparse
import time
import numpy as np

# Add project root to path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PROJECT_ROOT)

from utils.common.datagram import parse_npy_datagrams
from utils.common.lifter import CameraSpaceLifter


def main():
    parser = argparse.ArgumentParser(description='Lift 2D COCO17 keypoints to SMPL mesh from datagram npy file')
    parser.add_argument('npy_path', type=str, help='Path to npy file containing datagrams')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output path for results (npy format)')
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--window_size', '-w', type=int, default=16,
                        help='Sliding window size for temporal smoothing (default: 16)')
    parser.add_argument('--stride', '-s', type=int, default=1, help='Stride for sliding window (default: 1)')
    parser.add_argument('--ignore_confidence', action='store_true',
                        help='Ignore confidence values and treat all keypoints as valid')
    args = parser.parse_args()

    if not os.path.exists(args.npy_path):
        print(f"Error: File not found: {args.npy_path}")
        sys.exit(1)

    # Parse datagrams
    print(f"Loading datagrams from: {args.npy_path}")
    datagrams = parse_npy_datagrams(args.npy_path)
    print(f"Found {len(datagrams)} datagram(s)")

    if len(datagrams) == 0:
        print("Error: No datagrams found in the file")
        sys.exit(1)

    for i, dg in enumerate(datagrams):
        if 'pose_2d' not in dg:
            print(f"Error: Datagram {i} does not contain pose_2d (flags={dg['flags']})")
            sys.exit(1)

    # Determine device
    device = args.device
    if device == 'cuda' and not __import__('torch').cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    # Extract keypoints and confidence
    all_kp2d = np.stack([dg['pose_2d'] for dg in datagrams])

    if args.ignore_confidence:
        print("Ignoring confidence values as requested, treating all keypoints as valid.")
        all_confidence = np.ones((len(datagrams), 17))
    else:
        all_confidence = np.stack([dg.get('confidence_2d', np.ones(17)) for dg in datagrams])
        max_conf = all_confidence.max()
        if max_conf < 0.3:
            print(f"\n[WARNING] Maximum confidence value in data is {max_conf:.5f}, which is below the threshold of 0.3.")
            print("This causes all keypoints to be masked/zeroed out, leading to a static predicted pose.")
            print("Automatically ignoring the confidence mask and treating all keypoints as valid.")
            all_confidence = np.ones((len(datagrams), 17))

    image_width = datagrams[0]['image_width']
    image_height = datagrams[0]['image_height']

    # Initialize lifter
    lifter = CameraSpaceLifter(device=device, window_size=args.window_size)

    # Process with sliding window
    print(f"\nProcessing with sliding window (size={args.window_size}, stride={args.stride})...")
    start_time = time.time()

    all_results = lifter.lift_sequence(all_kp2d, all_confidence, image_width, image_height, stride=args.stride)

    elapsed = time.time() - start_time
    fps = len(datagrams) / elapsed if elapsed > 0 else 0
    print(f"\nProcessed {len(datagrams)} frames in {elapsed:.2f}s ({fps:.1f} FPS)")

    for i, result in enumerate(all_results[:3]):
        print(f"  Frame {i}: vertices={result['vertices'].shape}, joints={result['joints_3d'].shape}")

    # Save results
    if args.output is None:
        base = os.path.splitext(args.npy_path)[0]
        args.output = base + '_smpl_results.npy'

    output_data = {
        'vertices': np.concatenate([r['vertices'] for r in all_results], axis=0),
        'joints_3d': np.concatenate([r['joints_3d'] for r in all_results], axis=0),
        'betas': np.concatenate([r['betas'] for r in all_results], axis=0),
        'root_rot6d': np.stack([r['root_rot6d'] for r in all_results], axis=0),
        'body_pose_rot6d': np.stack([r['body_pose_rot6d'] for r in all_results], axis=0),
    }

    np.save(args.output, output_data)
    print(f"\nResults saved to: {args.output}")
    print(f"  Vertices: {output_data['vertices'].shape}")
    print(f"  3D Joints: {output_data['joints_3d'].shape}")
    print(f"  Betas: {output_data['betas'].shape}")
    print(f"  Root rot6d: {output_data['root_rot6d'].shape}")
    print(f"  Body pose rot6d: {output_data['body_pose_rot6d'].shape}")


if __name__ == '__main__':
    main()
