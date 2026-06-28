"""
Read byte array from npy file and convert to datagram, then use WHAM to lift the pose_2d to smpl coordinates.

Datagram layout:
  [flags (1 byte)][pose_3d (17*3*4 bytes)][image_width (4 bytes)][image_height (4 bytes)][pose_2d (17*2*4 bytes)][confidence_2d (17*4 bytes)]

Flags:
  bit 0: pose_3d included
  bit 1: pose_2d included
  bit 2: confidence_2d included

Usage:
    python file.py <path_to_npy_file> [--output <output_path>] [--window_size 16] [--stride 1]
"""

import os
import sys
import argparse
import struct
import time

import torch
import numpy as np
from collections import deque

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import get_cfg_defaults
from lib.models import build_network, build_body_model
from lib.utils.imutils import compute_cam_intrinsics
from lib.utils.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from lib.utils.kp_utils import root_centering
from lib.data.utils.normalizer import Normalizer


def parse_datagram(byte_array):
    """Parse a single datagram from a byte array.

    Returns:
        dict with keys: 'flags', 'pose_3d' (optional), 'image_width', 'image_height',
                        'pose_2d' (optional), 'confidence_2d' (optional)
    """
    offset = 0

    # Read flags (1 byte)
    flags = byte_array[offset]
    offset += 1

    result = {'flags': flags}

    # bit 0: pose_3d included
    if flags & 0x01:
        pose_3d = np.frombuffer(byte_array, dtype=np.float32, count=17 * 3, offset=offset).copy().reshape(17, 3)
        offset += 17 * 3 * 4
        result['pose_3d'] = pose_3d

    # image dimensions (always present after pose_3d block)
    image_width = struct.unpack_from('<f', byte_array, offset)[0]
    offset += 4
    image_height = struct.unpack_from('<f', byte_array, offset)[0]
    offset += 4
    result['image_width'] = image_width
    result['image_height'] = image_height

    # bit 1: pose_2d included
    if flags & 0x02:
        pose_2d = np.frombuffer(byte_array, dtype=np.float32, count=17 * 2, offset=offset).copy().reshape(17, 2)
        offset += 17 * 2 * 4
        result['pose_2d'] = pose_2d

    # bit 2: confidence_2d included
    if flags & 0x04:
        confidence_2d = np.frombuffer(byte_array, dtype=np.float32, count=17, offset=offset).copy()
        offset += 17 * 4
        result['confidence_2d'] = confidence_2d

    return result


def parse_npy_datagrams(npy_path):
    """Load npy file and parse all datagrams.

    The npy file is expected to contain a byte array (or array of byte arrays)
    where each element is a datagram.

    Returns:
        list of dicts, one per frame.
    """
    data = np.load(npy_path, allow_pickle=True)

    # Handle fixed-length byte strings (e.g. dtype |S417)
    if data.dtype.kind in ('S', 'U'):
        datagrams = []
        for item in data:
            if isinstance(item, bytes):
                datagrams.append(parse_datagram(item))
            elif isinstance(item, np.bytes_):
                datagrams.append(parse_datagram(bytes(item)))
            elif isinstance(item, str):
                datagrams.append(parse_datagram(item.encode('latin-1')))
        return datagrams

    # Handle different possible formats
    if data.dtype == object:
        # Array of objects (variable-length byte arrays)
        datagrams = []
        for item in data:
            if isinstance(item, np.ndarray) and item.dtype == np.uint8:
                datagrams.append(parse_datagram(item.tobytes()))
            elif isinstance(item, bytes):
                datagrams.append(parse_datagram(item))
        return datagrams
    elif data.dtype == np.uint8:
        if data.ndim == 1:
            # Single datagram
            return [parse_datagram(data.tobytes())]
        elif data.ndim == 2:
            # Multiple datagrams stacked as rows
            return [parse_datagram(data[i].tobytes()) for i in range(len(data))]
    else:
        raise ValueError(
            f"Unexpected npy dtype: {data.dtype}, shape: {data.shape}. "
            "Expected uint8 byte arrays."
        )


def prepare_cfg():
    """Prepare WHAM configuration for demo."""
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    return cfg


def get_neutral_smpl_init(smpl, device):
    """Get neutral SMPL initialization for the first frame.

    Returns:
        init_kp: (1, 1, 17*3 + 17*2) initial keypoint input
        init_smpl: (1, 1, 24*6) initial SMPL parameters in rotation-6d
        init_root: (1, 1, 6) initial root orientation in rotation-6d
    """
    # SMPL expects global_orient (B, 1, 3, 3) and body_pose (B, 23, 3, 3)
    init_global_orient = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
    init_body_pose = torch.eye(3, device=device).unsqueeze(0).repeat(1, 23, 1, 1)  # (1, 23, 3, 3)
    init_betas = torch.zeros(1, 10, device=device)  # (1, 10)

    init_output = smpl.get_output(
        global_orient=init_global_orient,
        body_pose=init_body_pose,
        betas=init_betas,
        pose2rot=False,
        return_full_pose=True
    )

    # init_output.joints is (1, 17, 3)
    # init_kp for motion encoder: [root_centered_3d_kps (17*3), norm_2d_kps_with_location_scale (17*2+3)]
    # Total: 51 + 37 = 88, matching WHAM's expected input dimension
    init_kp3d = root_centering(init_output.joints[:, :17], 'coco')  # (1, 17, 3)
    init_kp = torch.cat([
        init_kp3d.reshape(1, 1, -1),  # (1, 1, 17*3 = 51)
        torch.zeros(1, 1, 17 * 2 + 3, device=device)  # placeholder for norm 2D kps + location/scale
    ], dim=-1)  # (1, 1, 88)

    # full_pose is (1, 24, 3, 3) -> rotation-6d -> (1, 24*6)
    # But motion_decoder expects (B, 1, 24*6), so add dim
    init_smpl = matrix_to_rotation_6d(init_output.full_pose).unsqueeze(1)  # (1, 1, 24*6)
    # global_orient is (1, 1, 3, 3) -> rotation-6d -> (1, 1, 6) which is correct for trajectory_decoder
    init_root = matrix_to_rotation_6d(init_output.global_orient)  # (1, 1, 6)

    return init_kp, init_smpl, init_root


def compute_bbox_from_keypoints(kp2d):
    """Compute bounding box from 2D keypoints.

    Args:
        kp2d: (B, 17, 2) tensor

    Returns:
        bbox: (B, 3) tensor [cx, cy, scale/200]
    """
    x_min = kp2d[..., 0].min(dim=-1)[0]
    y_min = kp2d[..., 1].min(dim=-1)[0]
    x_max = kp2d[..., 0].max(dim=-1)[0]
    y_max = kp2d[..., 1].max(dim=-1)[0]

    cx = (x_max + x_min) / 2
    cy = (y_max + y_min) / 2
    w = x_max - x_min
    h = y_max - y_min
    scale = torch.stack([w, h], dim=-1).max(dim=-1)[0] * 1.2  # scale factor like WHAM

    bbox = torch.stack([cx, cy, scale / 200], dim=-1)
    return bbox


class SlidingWindowBuffer:
    """Sliding window buffer for real-time 2D keypoint sequences.

    Maintains a fixed-size window of normalized keypoints and masks,
    sliding forward as new frames arrive.
    """

    def __init__(self, window_size=16):
        self.window_size = window_size
        self.kp2d_buffer = deque(maxlen=window_size)
        self.mask_buffer = deque(maxlen=window_size)
        self.bbox_buffer = deque(maxlen=window_size)
        self.image_size = None

    def reset(self):
        """Clear the buffer."""
        self.kp2d_buffer.clear()
        self.mask_buffer.clear()
        self.bbox_buffer.clear()
        self.image_size = None

    def is_ready(self):
        """Check if buffer has at least 1 frame."""
        return len(self.kp2d_buffer) > 0

    def is_full(self):
        """Check if buffer is full."""
        return len(self.kp2d_buffer) >= self.window_size

    @property
    def length(self):
        return len(self.kp2d_buffer)

    def push(self, norm_kp2d, mask, bbox, image_width, image_height):
        """Add a new frame to the buffer.

        Args:
            norm_kp2d: (17*2 + 3,) normalized keypoints with location/scale
            mask: (17,) binary mask (1 = low confidence)
            bbox: (3,) bounding box [cx, cy, scale/200]
            image_width: image width
            image_height: image height
        """
        self.kp2d_buffer.append(norm_kp2d)
        self.mask_buffer.append(mask)
        self.bbox_buffer.append(bbox)
        self.image_size = (image_width, image_height)

    def get_window_tensors(self, device):
        """Get stacked tensors for the current window.

        Returns:
            x: (1, F, 17*2 + 3) normalized keypoints
            mask: (1, F, 17) mask
            bbox: (1, F, 3) bounding boxes
            image_width: image width
            image_height: image height
        """
        F = len(self.kp2d_buffer)
        x = torch.stack(list(self.kp2d_buffer), dim=0).unsqueeze(0).to(device)  # (1, F, 37)
        mask = torch.stack(list(self.mask_buffer), dim=0).unsqueeze(0).to(device)  # (1, F, 17)
        bbox = torch.stack(list(self.bbox_buffer), dim=0).unsqueeze(0).to(device)  # (1, F, 3)
        image_width, image_height = self.image_size

        return x, mask, bbox, image_width, image_height


class CameraSpaceLifter:
    """Lifts 2D COCO17 keypoints to SMPL mesh in camera space using WHAM's 3D lifting stages.

    Uses a sliding window approach for temporal consistency and real-time performance.
    This bypasses the 2D detection and SLAM modules, using only:
      - MotionEncoder (Stage 1): Encodes 2D keypoints into motion context
      - TrajectoryDecoder (Stage 2): Decodes root trajectory (identity for camera-space)
      - MotionDecoder (Stage 4): Decodes SMPL pose, shape, and camera parameters

    Usage for real-time streaming:
        lifter = CameraSpaceLifter(device='cuda', window_size=16)
        lifter.start_stream(image_width, image_height)

        for each frame:
            result = lifter.push_frame(kp2d, confidence)
            if result is not None:
                # Use result['vertices'], result['joints_3d'], etc.
    """

    def __init__(self, cfg=None, device='cuda', window_size=16):
        if cfg is None:
            cfg = prepare_cfg()

        # Override device in config BEFORE building network (build_network uses cfg.DEVICE)
        if isinstance(device, str):
            cfg.DEVICE = device
        else:
            cfg.DEVICE = str(device)

        self.cfg = cfg
        self.device = cfg.DEVICE
        self.window_size = window_size

        # Build body model (SMPL)
        self.smpl = build_body_model(self.device, batch_size=1)

        # Build WHAM network
        self.network = build_network(cfg, self.smpl)
        self.network.to(self.device)
        self.network.eval()

        # Normalizer for 2D keypoints
        self.keypoints_normalizer = Normalizer(cfg)

        # Sliding window buffer
        self.buffer = SlidingWindowBuffer(window_size=window_size)

        # Neutral SMPL initialization (cached)
        self.init_kp_template, self.init_smpl_template, self.init_root_template = \
            get_neutral_smpl_init(self.smpl, self.device)

        # Pre-compute camera intrinsics placeholder
        self._intrinsics_cache = {}

        print(f"CameraSpaceLifter initialized on device: {self.device}")
        print(f"  Window size: {window_size}")

    def normalize_keypoints(self, kp2d, image_width, image_height):
        """Normalize COCO17 2D keypoints using WHAM's bbox-based normalizer.

        Args:
            kp2d: (17, 2) numpy array of COCO17 keypoints in pixel coordinates
            image_width: image width in pixels
            image_height: image height in pixels

        Returns:
            norm_kp2d: (17*2 + 3,) normalized keypoints with location/scale
            bbox: (3,) bounding box [cx, cy, scale/200]
        """
        res = torch.tensor([image_width, image_height]).float()
        intrinsics = compute_cam_intrinsics(res)

        # Compute bbox from keypoints
        kp2d_tensor = torch.from_numpy(kp2d).float().unsqueeze(0)  # (1, 17, 2)
        bbox = compute_bbox_from_keypoints(kp2d_tensor)

        # Normalize
        norm_kp2d, bbox = self.keypoints_normalizer(
            kp2d_tensor, res, intrinsics, 224, 224, bbox
        )

        return norm_kp2d.squeeze(0), bbox.squeeze(0)

    def start_stream(self, image_width, image_height):
        """Initialize for a new streaming session.

        Args:
            image_width: width of input frames
            image_height: height of input frames
        """
        self.buffer.reset()
        # Pre-compute camera intrinsics for this resolution
        res = torch.tensor([image_width, image_height]).float()
        intrinsics = compute_cam_intrinsics(res).to(self.device)
        self._intrinsics_cache[(image_width, image_height)] = intrinsics
        # Set image size on buffer so push_frame can use it
        self.buffer.image_size = (image_width, image_height)
        print(f"Stream started: {image_width}x{image_height}")

    def push_frame(self, kp2d, confidence=None):
        """Push a new frame and get SMPL result if ready.

        This is the main real-time interface. Call this for each incoming frame.
        Returns None until the buffer has enough frames, then returns the SMPL
        result for the most recent frame.

        Args:
            kp2d: (17, 2) numpy array of COCO17 keypoints in pixel coordinates
            confidence: (17,) numpy array of confidence scores (can be None)

        Returns:
            dict with SMPL outputs or None if buffer not ready:
                - 'vertices': (6890, 3) mesh vertices in camera space
                - 'joints_3d': (17, 3) 3D joints in camera space
                - 'betas': (10,) shape parameters
                - 'poses_root_cam': root orientation in camera space
        """
        if confidence is None:
            confidence = np.ones(17)

        # Use the image size from the buffer (set on first frame)
        if self.buffer.image_size is None:
            raise RuntimeError("Call start_stream() before push_frame()")

        image_width, image_height = self.buffer.image_size

        # Create mask for low-confidence keypoints (must be bool for indexing)
        mask = torch.from_numpy(confidence < 0.3).bool()  # (17,)

        # Normalize keypoints
        norm_kp2d, bbox = self.normalize_keypoints(kp2d, image_width, image_height)

        # Push to buffer
        self.buffer.push(norm_kp2d, mask, bbox, image_width, image_height)

        # Run inference if we have enough frames
        if self.buffer.is_ready():
            return self._infer_window(image_width, image_height)

        return None

    @torch.no_grad()
    def _infer_window(self, image_width, image_height):
        """Run inference on the current sliding window.

        Args:
            image_width: image width
            image_height: image height

        Returns:
            dict with SMPL outputs for the last frame in the window
        """
        # Get window tensors
        x, mask_input, bbox, img_w, img_h = self.buffer.get_window_tensors(self.device)
        F = x.shape[1]

        # Build init_kp: replace the placeholder 2D part with first frame's full norm kps (includes location/scale)
        init_kp = self.init_kp_template.clone()
        init_kp[:, :, 17 * 3:] = x[:, :1, :]  # Use first frame's full norm kps (17*2+3 = 37 dims)

        # Camera angular velocity (zero for camera-space, no SLAM)
        cam_angvel = torch.zeros(1, F, 6, device=self.device)

        # Get camera intrinsics
        intrinsics = self._intrinsics_cache.get(
            (image_width, image_height),
            compute_cam_intrinsics(torch.tensor([image_width, image_height]).float()).to(self.device)
        )

        res = torch.tensor([image_width, image_height], device=self.device).float().unsqueeze(0)

        # Forward pass through WHAM network
        pred = self.network(
            x,
            (init_kp, self.init_smpl_template),
            img_features=None,
            mask=mask_input,
            init_root=self.init_root_template,
            cam_angvel=cam_angvel,
            return_y_up=True,
            refine_traj=False,
            cam_intrinsics=intrinsics.unsqueeze(0),  # (1, 1, 3, 3) -> broadcast to (1, F, 3, 3)
            bbox=bbox,
            res=res,
        )

        # Extract results for the LAST frame in the window.
        # Network output shapes (B=1, F frames):
        #   pred['verts_cam'] is (B*F, 6890, 3) - SMPL flattens B*F
        #   pred['poses_body'] is (B*F, 23, 3, 3) - SMPL flattens B*F
        #   pred['poses_root_cam'] is (B*F, 3, 3) - SMPL flattens B*F
        #   pred['betas'] is (B, F, 10) = (1, F, 10) - NOT flattened
        # Use [-1:] for SMPL-flattened outputs, [:, -1, :] for betas.
        last_body = pred['poses_body'][-1:]  # (1, 23, 3, 3)
        last_root = pred['poses_root_cam'][-1:]  # (1, 3, 3)
        last_betas = pred['betas'][:, -1, :]  # (1, 10)
        last_verts = pred['verts_cam'][-1:]  # (1, 6890, 3)

        results = {
            'vertices': last_verts.cpu().numpy(),  # (1, 6890, 3)
            'poses_body': last_body.cpu().numpy(),  # (1, 23, 3, 3)
            'betas': last_betas.cpu().numpy(),  # (1, 10)
            'poses_root_cam': last_root.cpu().numpy(),  # (1, 3, 3)
        }

        # Get 3D joints - re-run SMPL with the predicted pose
        if last_root.dim() == 3:
            last_root_smpl = last_root.unsqueeze(1)  # (1, 1, 3, 3) for SMPL
        else:
            last_root_smpl = last_root

        smpl_out = self.smpl.get_output(
            global_orient=last_root_smpl,
            body_pose=last_body,
            betas=last_betas,
            pose2rot=False,
            return_full_pose=True
        )
        results['joints_3d'] = smpl_out.joints.cpu().numpy()  # (1, N_joints, 3)

        return results

    @torch.no_grad()
    def lift(self, kp2d, confidence, image_width, image_height):
        """Single-frame lift (no temporal context). For offline/batch use.

        Args:
            kp2d: (17, 2) numpy array of COCO17 keypoints in pixel coordinates
            confidence: (17,) numpy array of confidence scores (can be None)
            image_width: image width in pixels
            image_height: image height in pixels

        Returns:
            dict with SMPL outputs in camera space
        """
        # Prepare confidence mask
        if confidence is None:
            confidence = np.ones(17)

        mask = torch.from_numpy(confidence < 0.3).bool()
        norm_kp2d, bbox = self.normalize_keypoints(kp2d, image_width, image_height)

        x = norm_kp2d.unsqueeze(0).unsqueeze(0).to(self.device)
        mask_input = mask.unsqueeze(0).unsqueeze(0).to(self.device)

        init_kp = self.init_kp_template.clone()
        init_kp[:, :, 17 * 3:] = x[:, :, :17 * 2]

        cam_angvel = torch.zeros(1, 1, 6, device=self.device)
        intrinsics = compute_cam_intrinsics(torch.tensor([image_width, image_height]).float()).to(self.device)
        res = torch.tensor([image_width, image_height], device=self.device).float().unsqueeze(0)

        pred = self.network(
            x,
            (init_kp, self.init_smpl_template),
            img_features=None,
            mask=mask_input,
            init_root=self.init_root_template,
            cam_angvel=cam_angvel,
            return_y_up=True,
            refine_traj=False,
            cam_intrinsics=intrinsics.unsqueeze(0),
            bbox=bbox.unsqueeze(0).unsqueeze(0).to(self.device),
            res=res,
        )

        # For single frame (F=1):
        #   pred['verts_cam'] is (1, 6890, 3), pred['poses_body'] is (1, 23, 3, 3)
        #   pred['poses_root_cam'] is (1, 3, 3), pred['betas'] is (1, 1, 10)
        last_body = pred['poses_body'][-1:]  # (1, 23, 3, 3)
        last_root = pred['poses_root_cam'][-1:]  # (1, 3, 3)
        last_betas = pred['betas'][:, -1, :]  # (1, 10)

        if last_root.dim() == 3:
            last_root_smpl = last_root.unsqueeze(1)  # (1, 1, 3, 3) for SMPL
        else:
            last_root_smpl = last_root

        results = {
            'vertices': pred['verts_cam'][-1:].cpu().numpy(),  # (1, 6890, 3)
            'poses_body': last_body.cpu().numpy(),  # (1, 23, 3, 3)
            'betas': last_betas.cpu().numpy(),  # (1, 10)
            'poses_root_cam': last_root.cpu().numpy(),  # (1, 3, 3)
        }

        # Get 3D joints by re-running SMPL with the predicted pose
        smpl_out = self.smpl.get_output(
            global_orient=last_root_smpl,
            body_pose=last_body,
            betas=last_betas,
            pose2rot=False,
            return_full_pose=True
        )
        results['joints_3d'] = smpl_out.joints.cpu().numpy()  # (1, N_joints, 3)

        return results

    @torch.no_grad()
    def lift_sequence(self, all_kp2d, all_confidence, image_width, image_height, stride=1):
        """Lift a sequence using sliding window for temporal consistency.

        Args:
            all_kp2d: (N, 17, 2) numpy array
            all_confidence: (N, 17) numpy array (can be None)
            image_width: image width
            image_height: image height
            stride: stride for sliding window (1 = every frame)

        Returns:
            list of dicts, one per frame
        """
        N = len(all_kp2d)
        if all_confidence is None:
            all_confidence = [None] * N

        # Reset and start stream
        self.start_stream(image_width, image_height)

        all_results = []
        for i in range(N):
            result = self.push_frame(all_kp2d[i], all_confidence[i])
            if result is not None:
                all_results.append(result)

        # Pad the beginning with single-frame results for frames before window was full
        if len(all_results) < N:
            pad_count = N - len(all_results)
            padded_results = []
            for i in range(pad_count):
                single_result = self.lift(all_kp2d[i], all_confidence[i], image_width, image_height)
                padded_results.append(single_result)
            all_results = padded_results + all_results

        return all_results


def main():
    parser = argparse.ArgumentParser(description='Lift 2D COCO17 keypoints to SMPL mesh using WHAM')
    parser.add_argument('npy_path', type=str, help='Path to npy file containing datagrams')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output path for results (npy format)')
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--window_size', '-w', type=int, default=16,
                        help='Sliding window size for temporal smoothing (default: 16)')
    parser.add_argument('--stride', '-s', type=int, default=1,
                        help='Stride for sliding window (default: 1)')
    parser.add_argument('--ignore_confidence', action='store_true',
                        help='Ignore the confidence values from datagrams and treat all keypoints as valid')
    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.npy_path):
        print(f"Error: File not found: {args.npy_path}")
        sys.exit(1)

    # Parse datagrams from npy file
    print(f"Loading datagrams from: {args.npy_path}")
    datagrams = parse_npy_datagrams(args.npy_path)
    print(f"Found {len(datagrams)} datagram(s)")

    if len(datagrams) == 0:
        print("Error: No datagrams found in the file")
        sys.exit(1)

    # Check that pose_2d is available
    for i, dg in enumerate(datagrams):
        if 'pose_2d' not in dg:
            print(f"Error: Datagram {i} does not contain pose_2d (flags={dg['flags']})")
            sys.exit(1)

    # Determine device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
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
            print("To override this, check your exporter or pass valid confidence values.")
            all_confidence = np.ones((len(datagrams), 17))

    image_width = datagrams[0]['image_width']
    image_height = datagrams[0]['image_height']

    # Initialize lifter with sliding window
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

    # Save as a structured numpy file
    output_data = {
        'vertices': np.concatenate([r['vertices'] for r in all_results], axis=0),  # (N, 6890, 3)
        'joints_3d': np.concatenate([r['joints_3d'] for r in all_results], axis=0),  # (N, 17, 3)
        'betas': np.concatenate([r['betas'] for r in all_results], axis=0),  # (N, 10)
        'poses_root_cam': np.concatenate([r['poses_root_cam'] for r in all_results], axis=0),
    }

    np.save(args.output, output_data)
    print(f"\nResults saved to: {args.output}")
    print(f"  Vertices: {output_data['vertices'].shape}")
    print(f"  3D Joints: {output_data['joints_3d'].shape}")
    print(f"  Betas: {output_data['betas'].shape}")


if __name__ == '__main__':
    main()