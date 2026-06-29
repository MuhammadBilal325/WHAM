"""Datagram parsing and serialization for the WHAM lifting pipeline.

Datagram layout (input):
  [flags (1 byte)][pose_3d (17*3*4 bytes)][image_width (4 bytes)][image_height (4 bytes)][pose_2d (17*2*4 bytes)][confidence_2d (17*4 bytes)]

Flags:
  bit 0: pose_3d included
  bit 1: pose_2d included
  bit 2: confidence_2d included

Output datagram layout (compact SMPL thetas + root):
  [root_rot6d (6 floats)][body_pose_rot6d (23*6=138 floats)][betas (10 floats)]
  Total: 6 + 138 + 10 = 154 floats = 616 bytes per frame
"""

import struct
import numpy as np


# ── Input datagram parsing ──────────────────────────────────────────────────

def parse_input_datagram(byte_array):
    """Parse a single input datagram from a byte array.

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
                datagrams.append(parse_input_datagram(item))
            elif isinstance(item, np.bytes_):
                datagrams.append(parse_input_datagram(bytes(item)))
            elif isinstance(item, str):
                datagrams.append(parse_input_datagram(item.encode('latin-1')))
        return datagrams

    # Handle different possible formats
    if data.dtype == object:
        # Array of objects (variable-length byte arrays)
        datagrams = []
        for item in data:
            if isinstance(item, np.ndarray) and item.dtype == np.uint8:
                datagrams.append(parse_input_datagram(item.tobytes()))
            elif isinstance(item, bytes):
                datagrams.append(parse_input_datagram(item))
        return datagrams
    elif data.dtype == np.uint8:
        if data.ndim == 1:
            # Single datagram
            return [parse_input_datagram(data.tobytes())]
        elif data.ndim == 2:
            # Multiple datagrams stacked as rows
            return [parse_input_datagram(data[i].tobytes()) for i in range(len(data))]
    else:
        raise ValueError(
            f"Unexpected npy dtype: {data.dtype}, shape: {data.shape}. "
            "Expected uint8 byte arrays."
        )


# ── Output datagram serialization ──────────────────────────────────────────

# Output format: [root_rot6d (6 floats)][body_pose_rot6d (23*6=138 floats)][betas (10 floats)]
# Total floats: 6 + 138 + 10 = 154 floats = 616 bytes
OUTPUT_FLOATS = 154
OUTPUT_BYTES = OUTPUT_FLOATS * 4  # 616


def serialize_output_datagram(root_rot6d, body_pose_rot6d, betas):
    """Serialize SMPL output into a compact binary datagram.

    Args:
        root_rot6d: (6,) numpy array - root orientation in rotation-6d
        body_pose_rot6d: (138,) numpy array - 23 body joints in rotation-6d (23*6)
        betas: (10,) numpy array - SMPL shape parameters

    Returns:
        bytes: compact binary datagram (616 bytes)
    """
    buf = np.concatenate([
        root_rot6d.flatten(),           # 6
        body_pose_rot6d.flatten(),      # 138
        betas.flatten(),                # 10
    ]).astype(np.float32)
    return buf.tobytes()


def deserialize_output_datagram(data_bytes):
    """Deserialize a compact SMPL output datagram.

    Args:
        data_bytes: bytes (616 bytes)

    Returns:
        dict with keys: 'root_rot6d' (6,), 'body_pose_rot6d' (138,), 'betas' (10,)
    """
    arr = np.frombuffer(data_bytes, dtype=np.float32)
    return {
        'root_rot6d': arr[0:6],
        'body_pose_rot6d': arr[6:144],
        'betas': arr[144:154],
    }
