"""Visualization utilities for SMPL mesh rendering.

Provides functions to render SMPL meshes and skeletons to images/video
using OpenCV with simple orthographic projection.
"""

import os
import sys
import numpy as np
import cv2

# COCO17 skeleton connections for visualization
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),            # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]

SKELETON_COLOR = (0, 0, 255)    # red (BGR)
JOINT_COLOR = (255, 0, 0)       # blue (BGR)


def project_vertices_to_2d(vertices, image_size, scale_factor=0.4):
    """Project 3D vertices to 2D using simple orthographic projection.

    Args:
        vertices: (V, 3) array of vertices in camera space
        image_size: output image size (square)
        scale_factor: fraction of image to use

    Returns:
        pts_2d: (V, 2) array of 2D pixel coordinates
        depth: (V,) array of depth values (Z) for sorting
    """
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    v_center = (v_min + v_max) / 2
    v_scale = (v_max - v_min).max()
    if v_scale < 1e-6:
        v_scale = 1.0

    pts_2d = np.zeros((len(vertices), 2))
    pts_2d[:, 0] = (vertices[:, 0] - v_center[0]) / v_scale * image_size * scale_factor + image_size / 2
    pts_2d[:, 1] = (vertices[:, 1] - v_center[1]) / v_scale * image_size * scale_factor + image_size / 2

    depth = vertices[:, 2]
    return pts_2d, depth


def render_mesh_wireframe(img, vertices, faces, image_size, color=(200, 200, 200)):
    """Render a wireframe mesh onto the image."""
    pts_2d, depth = project_vertices_to_2d(vertices, image_size)

    for face in faces:
        pts = pts_2d[face].astype(np.int32)
        if np.all(pts >= 0) and np.all(pts < image_size):
            cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)

    return img


def render_mesh_solid(img, vertices, faces, image_size, color=(180, 180, 220)):
    """Render a solid (flat-shaded) mesh onto the image with depth sorting."""
    pts_2d, depth = project_vertices_to_2d(vertices, image_size)

    # Compute face normals vectorized
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norm_lens = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_lens = np.maximum(norm_lens, 1e-8)
    normals = normals / norm_lens

    # Compute simple face shading vectorized
    light_dir = np.array([0.0, 0.5, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)
    shades = np.dot(normals, light_dir)
    shades = np.maximum(0.3, shades)

    # Compute face colors
    face_colors = (shades[:, None] * np.array(color)).astype(np.uint8)

    # Compute face depths (average Z of vertices) for sorting
    face_depths = np.mean(depth[faces], axis=1)
    sorted_indices = np.argsort(-face_depths)

    for idx in sorted_indices:
        face = faces[idx]
        pts = pts_2d[face].astype(np.int32)

        if np.all(pts >= -image_size) and np.all(pts < 2 * image_size):
            cv2.fillConvexPoly(img, pts, tuple(map(int, face_colors[idx])), cv2.LINE_AA)

    return img


def render_skeleton(img, joints_3d, image_size):
    """Render COCO17 skeleton onto the image."""
    pts_2d, _ = project_vertices_to_2d(joints_3d, image_size)
    n_joints = len(pts_2d)

    for (a, b) in COCO_SKELETON:
        if a < n_joints and b < n_joints:
            pt1 = tuple(pts_2d[a].astype(int))
            pt2 = tuple(pts_2d[b].astype(int))
            cv2.line(img, pt1, pt2, SKELETON_COLOR, 2, cv2.LINE_AA)

    for j in range(n_joints):
        pt = tuple(pts_2d[j].astype(int))
        cv2.circle(img, pt, 4, JOINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(img, pt, 4, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def reshape_data(data):
    """Reshape vertices and joints from output into proper (N_frames, N_verts, 3) format.

    Returns:
        vertices: (N_frames, N_verts, 3) or None if vertices are unusable
        joints_3d: (N_frames, N_joints, 3) or None
        n_frames: number of frames
    """
    vertices = data['vertices']
    joints_3d = data.get('joints_3d', None)

    SMPL_NUM_VERTS = 6890

    n_frames_from_joints = len(joints_3d) if joints_3d is not None and joints_3d.ndim >= 2 else None

    if vertices.ndim == 3 and vertices.shape[1] == SMPL_NUM_VERTS:
        return vertices, joints_3d, vertices.shape[0]

    if vertices.ndim == 2 and vertices.shape[0] % SMPL_NUM_VERTS == 0:
        n_frames = vertices.shape[0] // SMPL_NUM_VERTS
        vertices = vertices.reshape(n_frames, SMPL_NUM_VERTS, 3)
        return vertices, joints_3d, n_frames

    if vertices.ndim == 3:
        return vertices, joints_3d, vertices.shape[0]

    if vertices.ndim == 2 and n_frames_from_joints is not None:
        if vertices.shape[0] % n_frames_from_joints == 0:
            n_verts_per_frame = vertices.shape[0] // n_frames_from_joints
            vertices = vertices.reshape(n_frames_from_joints, n_verts_per_frame, 3)
            return vertices, joints_3d, n_frames_from_joints

    print(f"  WARNING: vertices shape {vertices.shape} is inconsistent.")
    print(f"  Attempting to use joints_3d for skeleton-only visualization.")
    if n_frames_from_joints is not None:
        print(f"  Using {n_frames_from_joints} frames from joints_3d.")
        return None, joints_3d, n_frames_from_joints

    raise ValueError(
        f"Cannot reshape vertices with shape {vertices.shape}. "
        f"Total vertices ({vertices.shape[0]}) is not divisible by "
        f"SMPL vertex count ({SMPL_NUM_VERTS})."
    )


def visualize_smpl_results(data, output_path, device='cpu', fps=30, image_size=512, mode='mesh'):
    """Visualize SMPL mesh results from the output data.

    Args:
        data: dict with keys 'vertices' (N, 6890, 3), 'joints_3d' (N, 17, 3),
              'betas' (N, 10), 'poses_root_cam' (N, 3, 3)
        output_path: path to save the output video
        device: 'cuda' or 'cpu' (only used for loading body model)
        fps: frames per second for output video
        image_size: size of the output frames (square)
        mode: 'mesh', 'wireframe', 'skeleton', or 'mesh+skeleton'
    """
    # From src/utils/common/visualizer.py -> WHAM/ is 3 levels up
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from lib.models import build_body_model

    vertices, joints_3d, N = reshape_data(data)

    print(f"Visualizing {N} frames...")
    if vertices is not None:
        print(f"  Vertices shape: {vertices.shape}")
    else:
        print(f"  Vertices: unavailable (using skeleton only)")
    if joints_3d is not None:
        print(f"  Joints shape: {joints_3d.shape}")

    faces = None
    if vertices is not None and mode in ('mesh', 'mesh+skeleton', 'wireframe'):
        smpl = build_body_model(device, batch_size=1)
        faces = smpl.faces
        print(f"  Faces shape: {faces.shape}")

    if vertices is None and mode in ('mesh', 'mesh+skeleton', 'wireframe'):
        print(f"  WARNING: Mesh data unavailable, falling back to skeleton-only mode")
        mode = 'skeleton'

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (image_size, image_size))

    if not writer.isOpened():
        print("Error: Could not open video writer")
        sys.exit(1)

    for i in range(N):
        background = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

        if mode in ('mesh', 'mesh+skeleton') and vertices is not None and faces is not None:
            verts = vertices[i]
            img = render_mesh_solid(background, verts, faces, image_size, color=(180, 180, 220))
        elif mode == 'wireframe' and vertices is not None and faces is not None:
            verts = vertices[i]
            img = render_mesh_wireframe(background, verts, faces, image_size)
        else:
            img = background.copy()

        if mode in ('skeleton', 'mesh+skeleton') and joints_3d is not None:
            joints = joints_3d[i]
            img = render_skeleton(img, joints, image_size)

        cv2.putText(img, f'Frame: {i}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        writer.write(img)

        if (i + 1) % 10 == 0 or i == N - 1:
            print(f"  Rendered frame {i + 1}/{N}", end='\r')

    writer.release()
    print(f"\nVideo saved to: {output_path}")
