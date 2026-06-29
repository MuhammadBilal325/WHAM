"""WHAM-based 2D-to-3D SMPL lifting engine.

This module provides the CameraSpaceLifter class that wraps WHAM for
lifting 2D COCO17 keypoints to SMPL mesh parameters in camera space.

Designed to be backend-agnostic: swap PyTorch for ONNX by subclassing
and overriding _infer_window() and lift().
"""

import os
import sys
import numpy as np
import torch
from collections import deque

# Ensure project root is on path for config imports
# From src/utils/common/lifter.py -> WHAM/ is 3 levels up
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs.config import get_cfg_defaults
from lib.models import build_network, build_body_model
from lib.utils.imutils import compute_cam_intrinsics
from lib.utils.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from lib.utils.kp_utils import root_centering
from lib.data.utils.normalizer import Normalizer


def _compute_bbox_from_keypoints(kp2d):
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
    scale = torch.stack([w, h], dim=-1).max(dim=-1)[0] * 1.2

    bbox = torch.stack([cx, cy, scale / 200], dim=-1)
    return bbox


class _SlidingWindowBuffer:
    """Sliding window buffer for real-time 2D keypoint sequences."""

    def __init__(self, window_size=16):
        self.window_size = window_size
        self.kp2d_buffer = deque(maxlen=window_size)
        self.mask_buffer = deque(maxlen=window_size)
        self.bbox_buffer = deque(maxlen=window_size)
        self.image_size = None

    def reset(self):
        self.kp2d_buffer.clear()
        self.mask_buffer.clear()
        self.bbox_buffer.clear()
        self.image_size = None

    def is_ready(self):
        return len(self.kp2d_buffer) > 0

    def is_full(self):
        return len(self.kp2d_buffer) >= self.window_size

    @property
    def length(self):
        return len(self.kp2d_buffer)

    def push(self, norm_kp2d, mask, bbox, image_width, image_height):
        self.kp2d_buffer.append(norm_kp2d)
        self.mask_buffer.append(mask)
        self.bbox_buffer.append(bbox)
        self.image_size = (image_width, image_height)

    def get_window_tensors(self, device):
        F = len(self.kp2d_buffer)
        x = torch.stack(list(self.kp2d_buffer), dim=0).unsqueeze(0).to(device)
        mask = torch.stack(list(self.mask_buffer), dim=0).unsqueeze(0).to(device)
        bbox = torch.stack(list(self.bbox_buffer), dim=0).unsqueeze(0).to(device)
        image_width, image_height = self.image_size
        return x, mask, bbox, image_width, image_height


class CameraSpaceLifter:
    """Lifts 2D COCO17 keypoints to SMPL mesh in camera space using WHAM.

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
            cfg = get_cfg_defaults()
            cfg.merge_from_file(os.path.join(_PROJECT_ROOT, 'configs', 'yamls', 'demo.yaml'))

        # Override device in config BEFORE building network
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
        self.buffer = _SlidingWindowBuffer(window_size=window_size)

        # Neutral SMPL initialization (cached)
        self.init_kp_template, self.init_smpl_template, self.init_root_template = \
            self._get_neutral_smpl_init()

        # Pre-compute camera intrinsics placeholder
        self._intrinsics_cache = {}

        print(f"CameraSpaceLifter initialized on device: {self.device}")
        print(f"  Window size: {window_size}")

    def _get_neutral_smpl_init(self):
        """Get neutral SMPL initialization for the first frame."""
        init_global_orient = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0)
        init_body_pose = torch.eye(3, device=self.device).unsqueeze(0).repeat(1, 23, 1, 1)
        init_betas = torch.zeros(1, 10, device=self.device)

        init_output = self.smpl.get_output(
            global_orient=init_global_orient,
            body_pose=init_body_pose,
            betas=init_betas,
            pose2rot=False,
            return_full_pose=True
        )

        init_kp3d = root_centering(init_output.joints[:, :17], 'coco')
        init_kp = torch.cat([
            init_kp3d.reshape(1, 1, -1),
            torch.zeros(1, 1, 17 * 2 + 3, device=self.device)
        ], dim=-1)

        init_smpl = matrix_to_rotation_6d(init_output.full_pose).unsqueeze(1)
        init_root = matrix_to_rotation_6d(init_output.global_orient)

        return init_kp, init_smpl, init_root

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

        kp2d_tensor = torch.from_numpy(kp2d).float().unsqueeze(0)
        bbox = _compute_bbox_from_keypoints(kp2d_tensor)

        norm_kp2d, bbox = self.keypoints_normalizer(
            kp2d_tensor, res, intrinsics, 224, 224, bbox
        )

        return norm_kp2d.squeeze(0), bbox.squeeze(0)

    def start_stream(self, image_width, image_height):
        """Initialize for a new streaming session."""
        self.buffer.reset()
        res = torch.tensor([image_width, image_height]).float()
        intrinsics = compute_cam_intrinsics(res).to(self.device)
        self._intrinsics_cache[(image_width, image_height)] = intrinsics
        self.buffer.image_size = (image_width, image_height)
        print(f"Stream started: {image_width}x{image_height}")

    def push_frame(self, kp2d, confidence=None):
        """Push a new frame and get SMPL result if ready.

        Args:
            kp2d: (17, 2) numpy array of COCO17 keypoints in pixel coordinates
            confidence: (17,) numpy array of confidence scores (can be None)

        Returns:
            dict with SMPL outputs or None if buffer not ready:
                - 'vertices': (6890, 3) mesh vertices in camera space
                - 'joints_3d': (17, 3) 3D joints in camera space
                - 'betas': (10,) shape parameters
                - 'root_rot6d': (6,) root orientation in rotation-6d
                - 'body_pose_rot6d': (138,) body pose in rotation-6d
        """
        if confidence is None:
            confidence = np.ones(17)

        if self.buffer.image_size is None:
            raise RuntimeError("Call start_stream() before push_frame()")

        image_width, image_height = self.buffer.image_size

        mask = torch.from_numpy(confidence < 0.3).bool()
        norm_kp2d, bbox = self.normalize_keypoints(kp2d, image_width, image_height)

        self.buffer.push(norm_kp2d, mask, bbox, image_width, image_height)

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
        x, mask_input, bbox, img_w, img_h = self.buffer.get_window_tensors(self.device)
        F = x.shape[1]

        init_kp = self.init_kp_template.clone()
        init_kp[:, :, 17 * 3:] = x[:, :1, :]

        cam_angvel = torch.zeros(1, F, 6, device=self.device)

        intrinsics = self._intrinsics_cache.get(
            (image_width, image_height),
            compute_cam_intrinsics(torch.tensor([image_width, image_height]).float()).to(self.device)
        )

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
            bbox=bbox,
            res=res,
        )

        last_body = pred['poses_body'][-1:]  # (1, 23, 3, 3)
        last_root = pred['poses_root_cam'][-1:]  # (1, 3, 3)
        last_betas = pred['betas'][:, -1, :]  # (1, 10)
        last_verts = pred['verts_cam'][-1:]  # (1, 6890, 3)

        # Convert to rotation-6d for compact output
        root_rot6d = matrix_to_rotation_6d(last_root).squeeze(0).cpu().numpy()  # (6,)
        body_pose_rot6d = matrix_to_rotation_6d(last_body).squeeze(0).cpu().numpy()  # (138,)

        results = {
            'vertices': last_verts.cpu().numpy(),
            'betas': last_betas.cpu().numpy(),
            'root_rot6d': root_rot6d,
            'body_pose_rot6d': body_pose_rot6d,
        }

        # Get 3D joints
        if last_root.dim() == 3:
            last_root_smpl = last_root.unsqueeze(1)
        else:
            last_root_smpl = last_root

        smpl_out = self.smpl.get_output(
            global_orient=last_root_smpl,
            body_pose=last_body,
            betas=last_betas,
            pose2rot=False,
            return_full_pose=True
        )
        results['joints_3d'] = smpl_out.joints.cpu().numpy()

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

        last_body = pred['poses_body'][-1:]
        last_root = pred['poses_root_cam'][-1:]
        last_betas = pred['betas'][:, -1, :]

        if last_root.dim() == 3:
            last_root_smpl = last_root.unsqueeze(1)
        else:
            last_root_smpl = last_root

        root_rot6d = matrix_to_rotation_6d(last_root).squeeze(0).cpu().numpy()
        body_pose_rot6d = matrix_to_rotation_6d(last_body).squeeze(0).cpu().numpy()

        results = {
            'vertices': pred['verts_cam'][-1:].cpu().numpy(),
            'betas': last_betas.cpu().numpy(),
            'root_rot6d': root_rot6d,
            'body_pose_rot6d': body_pose_rot6d,
        }

        smpl_out = self.smpl.get_output(
            global_orient=last_root_smpl,
            body_pose=last_body,
            betas=last_betas,
            pose2rot=False,
            return_full_pose=True
        )
        results['joints_3d'] = smpl_out.joints.cpu().numpy()

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
