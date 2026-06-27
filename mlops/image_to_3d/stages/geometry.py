from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from mlops.image_to_3d.scene import Scene


def run(
    scene: Scene,
    image_path: Path,
    depth_path: Path,
    intrinsics_path: Path,
    *,
    mesh_stride: int = 2,
    max_depth: float = 4.0,
) -> tuple[Path, Path]:
    depth = np.load(depth_path).astype("float32")
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype="uint8")
    intr = json.loads(Path(intrinsics_path).read_text(encoding="utf-8"))
    points, colors = _depth_to_points(depth, rgb, intr, max_depth=max_depth)
    points_path = Path(scene.root) / "points.ply"
    _write_ply(points_path, points, colors)
    mesh_path = Path(scene.root) / "mesh.glb"
    _write_grid_mesh(mesh_path, depth, rgb, intr, stride=max(1, int(mesh_stride)), max_depth=max_depth)
    scene.add_artifact(
        points_path,
        kind="point_cloud",
        source="image_depth",
        confidence=0.58,
        stage_version="rgbd-point-cloud@v1",
        depends_on=["depth.npy", "intrinsics.json", Path(image_path).name],
    )
    scene.add_artifact(
        mesh_path,
        kind="mesh",
        source="image_depth",
        confidence=0.52,
        stage_version="rgbd-grid-mesh@v1",
        depends_on=["depth.npy", "intrinsics.json", Path(image_path).name],
    )
    return points_path, mesh_path


def _depth_to_points(
    depth: np.ndarray,
    rgb: np.ndarray,
    intr: dict[str, float],
    *,
    max_depth: float,
    sample_limit: int | None = 250_000,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]
    z = 0.3 + (1.0 - depth) * float(max_depth)
    x = (xs.astype("float32") - float(intr["cx"])) * z / float(intr["fx"])
    y = -(ys.astype("float32") - float(intr["cy"])) * z / float(intr["fy"])
    points = np.stack([x, y, -z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3)
    if sample_limit is None:
        return points.astype("float32"), colors.astype("uint8")
    sample = max(1, int(len(points) / int(sample_limit)))
    return points[::sample].astype("float32"), colors[::sample].astype("uint8")


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    try:
        import open3d as o3d  # type: ignore[import-not-found]

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points.astype("float64"))
        pc.colors = o3d.utility.Vector3dVector((colors.astype("float64") / 255.0))
        o3d.io.write_point_cloud(str(path), pc, write_ascii=False)
        return
    except Exception:
        pass
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def _write_grid_mesh(
    path: Path,
    depth: np.ndarray,
    rgb: np.ndarray,
    intr: dict[str, float],
    *,
    stride: int,
    max_depth: float,
) -> None:
    d = depth[::stride, ::stride]
    c = rgb[::stride, ::stride]
    points, colors = _depth_to_points(
        d,
        c,
        _scaled_intrinsics(intr, stride),
        max_depth=max_depth,
        sample_limit=None,
    )
    hh, ww = d.shape
    faces: list[list[int]] = []
    for y in range(hh - 1):
        row = y * ww
        next_row = (y + 1) * ww
        for x in range(ww - 1):
            a = row + x
            b = row + x + 1
            cidx = next_row + x
            didx = next_row + x + 1
            faces.append([a, cidx, b])
            faces.append([b, cidx, didx])
    face_array = np.asarray(faces, dtype="uint32")
    try:
        import trimesh  # type: ignore[import-not-found]

        mesh = trimesh.Trimesh(vertices=points, faces=face_array, vertex_colors=colors, process=False)
        mesh.export(path)
    except Exception:
        _write_simple_glb(path, points, face_array, colors)


def _scaled_intrinsics(intr: dict[str, float], stride: int) -> dict[str, float]:
    return {
        "fx": float(intr["fx"]) / stride,
        "fy": float(intr["fy"]) / stride,
        "cx": float(intr["cx"]) / stride,
        "cy": float(intr["cy"]) / stride,
    }


def _write_simple_glb(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    vertices = np.asarray(vertices, dtype="<f4")
    indices = np.asarray(faces.reshape(-1), dtype="<u4")
    rgba = np.zeros((len(colors), 4), dtype="uint8")
    rgba[:, :3] = np.asarray(colors, dtype="uint8")
    rgba[:, 3] = 255

    chunks: list[bytes] = []
    views: list[dict[str, int]] = []
    offset = 0
    for data, target in [
        (vertices.tobytes(), 34962),
        (rgba.tobytes(), 34962),
        (indices.tobytes(), 34963),
    ]:
        padded = _pad4(data)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target})
        chunks.append(padded)
        offset += len(padded)
    body = b"".join(chunks)
    mins = vertices.min(axis=0).astype(float).tolist() if len(vertices) else [0.0, 0.0, 0.0]
    maxs = vertices.max(axis=0).astype(float).tolist() if len(vertices) else [0.0, 0.0, 0.0]
    gltf = {
        "asset": {"version": "2.0", "generator": "cvLayer image_to_3d"},
        "buffers": [{"byteLength": len(body)}],
        "bufferViews": views,
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": int(len(vertices)),
                "type": "VEC3",
                "min": mins,
                "max": maxs,
            },
            {
                "bufferView": 1,
                "componentType": 5121,
                "count": int(len(rgba)),
                "type": "VEC4",
                "normalized": True,
            },
            {
                "bufferView": 2,
                "componentType": 5125,
                "count": int(len(indices)),
                "type": "SCALAR",
            },
        ],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "roughnessFactor": 1}}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "COLOR_0": 1},
                        "indices": 2,
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        ],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    json_bytes = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), pad=b" ")
    total = 12 + 8 + len(json_bytes) + 8 + len(body)
    with path.open("wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(json_bytes), b"JSON"))
        f.write(json_bytes)
        f.write(struct.pack("<I4s", len(body), b"BIN\x00"))
        f.write(body)


def _pad4(data: bytes, pad: bytes = b"\x00") -> bytes:
    rem = len(data) % 4
    if rem == 0:
        return data
    return data + pad * (4 - rem)
