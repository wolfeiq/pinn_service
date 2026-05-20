"""ROS 2 bag ingestion via the ``rosbags`` library.

Reads a rosbag2 directory (``.db3`` with metadata.yaml) and converts
the messages we care about into the engine's unified sensor schema:
``{sensor_name: (timestamps, observations)}``.

Three sensor types covered out of the box (the build plan's list):

* ``sensor_msgs/Imu`` — extract ``angular_velocity`` or
  ``linear_acceleration`` (per ``field``).
* ``sensor_msgs/JointState`` — extract ``position[i]`` / ``velocity[i]``
  / ``effort[i]`` for a named joint.
* ``geometry_msgs/WrenchStamped`` — extract ``wrench.force.<axis>`` or
  ``wrench.torque.<axis>``.
* ``nav_msgs/Odometry`` — extract ``pose.pose.position.<axis>`` or
  ``twist.twist.linear.<axis>``.

``rosbags`` is intentionally an *optional* dependency — it's only
imported when this module is used, so users without ROS installed can
still use the rest of the engine.

Usage::

    from pinn_engine.data.rosbag import load_ros_bag

    data = load_ros_bag(
        path="/path/to/rosbag2_dir",
        topic_mapping={
            "u_meas": {"topic": "/odom", "field": "twist.twist.linear.x"},
            "imu_az": {"topic": "/imu",  "field": "linear_acceleration.z"},
        },
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


class RosBagDepError(ImportError):
    """Raised when ``rosbags`` is not installed."""


def _require_rosbags():
    try:
        import rosbags  # noqa: F401
    except ImportError as e:
        raise RosBagDepError(
            "ROS bag ingestion requires the `rosbags` package. "
            "Install with: pip install rosbags"
        ) from e


def _attr_path(obj: Any, path: str) -> Any:
    """Resolve a dotted-name path on a message object.

    Supports indexing via ``[i]`` suffix on the last component for arrays:
    e.g. ``position[2]`` reads ``msg.position[2]``.
    """
    parts = path.split(".")
    cur = obj
    for part in parts:
        idx = None
        if "[" in part and part.endswith("]"):
            base, idx_str = part[:-1].split("[", 1)
            idx = int(idx_str)
            part = base
        cur = getattr(cur, part)
        if idx is not None:
            cur = cur[idx]
    return cur


def load_ros_bag(
    path: str | Path,
    topic_mapping: Dict[str, Dict[str, str]],
    t_zero: float | None = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Read a rosbag2 directory and emit our unified sensor schema.

    Parameters:
        path: Path to the rosbag2 directory (containing ``metadata.yaml``).
        topic_mapping: ``{sensor_name: {"topic": "/foo", "field": "bar.baz"}}``.
            ``sensor_name`` is what your :class:`Sensor` declarations use;
            ``topic`` is the ROS topic; ``field`` is the dotted-name path
            into the message (supports ``[i]`` indexing).
        t_zero: optional timestamp (in seconds) to subtract from all
            recorded times. Default: use the earliest timestamp seen.

    Returns:
        ``{sensor_name: (t_array, value_array)}``, both numpy float32.
    """
    _require_rosbags()
    from rosbags.rosbag2 import Reader
    from rosbags.serde import deserialize_cdr

    path = Path(path)
    if not (path / "metadata.yaml").exists():
        raise FileNotFoundError(
            f"{path} doesn't look like a rosbag2 directory (no metadata.yaml)"
        )

    # Buckets: {sensor_name: ([t,...], [val,...])}
    buckets: Dict[str, Tuple[list, list]] = {
        name: ([], []) for name in topic_mapping
    }

    with Reader(str(path)) as reader:
        topics = {c.topic: c for c in reader.connections}
        # Map sensor_name -> connection (for fast filter)
        wanted = {}
        for name, spec in topic_mapping.items():
            topic = spec["topic"]
            if topic not in topics:
                raise KeyError(
                    f"Topic {topic!r} (for sensor {name!r}) not in bag. "
                    f"Available: {list(topics.keys())}"
                )
            wanted[topics[topic].id] = (name, spec["field"], topics[topic].msgtype)

        for connection, timestamp, raw in reader.messages():
            entry = wanted.get(connection.id)
            if entry is None:
                continue
            sensor_name, field, msgtype = entry
            msg = deserialize_cdr(raw, msgtype)
            value = _attr_path(msg, field)
            # rosbag timestamps are nanoseconds; convert to seconds.
            buckets[sensor_name][0].append(timestamp / 1e9)
            buckets[sensor_name][1].append(float(value))

    # Time zeroing.
    if t_zero is None:
        all_starts = [bs[0][0] for bs in buckets.values() if bs[0]]
        t_zero = min(all_starts) if all_starts else 0.0

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, (ts, vs) in buckets.items():
        t_arr = np.asarray(ts, dtype=np.float32) - t_zero
        v_arr = np.asarray(vs, dtype=np.float32)
        out[name] = (t_arr, v_arr)
    return out


__all__ = ["load_ros_bag", "RosBagDepError"]
