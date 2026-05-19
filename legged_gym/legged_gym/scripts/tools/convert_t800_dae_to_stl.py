#!/usr/bin/env python3
"""Generate Isaac Gym friendly STL meshes for the T800 URDF.

The source T800 assets are COLLADA files. EngineAI's public SA01 Isaac Gym
example ships STL meshes, so this script creates an STL-backed T800 URDF while
preserving the simple one-mesh-per-link layout.
"""

from __future__ import annotations

import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


COLLADA_NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
LEGGED_GYM_ROOT_DIR = Path(__file__).resolve().parents[3]
SOURCE_ROBOT_DIR = LEGGED_GYM_ROOT_DIR / "resources" / "robots" / "t800"
TARGET_ROBOT_DIR = LEGGED_GYM_ROOT_DIR / "resources" / "robots" / "t800_stl"


Vector3 = Tuple[float, float, float]
Matrix4 = Tuple[Tuple[float, float, float, float], ...]


IDENTITY_MATRIX: Matrix4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _strip_fragment(value: str) -> str:
    return value[1:] if value.startswith("#") else value


def _parse_float_array(text: str) -> List[float]:
    return [float(item) for item in text.split()]


def _parse_matrix(text: str | None) -> Matrix4:
    if not text:
        return IDENTITY_MATRIX
    values = _parse_float_array(text)
    if len(values) != 16:
        raise ValueError(f"Expected 16 matrix values, got {len(values)}")
    return (
        tuple(values[0:4]),
        tuple(values[4:8]),
        tuple(values[8:12]),
        tuple(values[12:16]),
    )


def _matrix_determinant_3x3(matrix: Matrix4) -> float:
    a, b, c = matrix[0][:3]
    d, e, f = matrix[1][:3]
    g, h, i = matrix[2][:3]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _transform_point(point: Vector3, matrix: Matrix4) -> Vector3:
    x, y, z = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    )


def _vector_sub(lhs: Vector3, rhs: Vector3) -> Vector3:
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])


def _cross(lhs: Vector3, rhs: Vector3) -> Vector3:
    return (
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    )


def _normal(triangle: Sequence[Vector3]) -> Vector3:
    edge_a = _vector_sub(triangle[1], triangle[0])
    edge_b = _vector_sub(triangle[2], triangle[0])
    normal = _cross(edge_a, edge_b)
    length = math.sqrt(sum(component * component for component in normal))
    if length == 0.0:
        return (0.0, 0.0, 0.0)
    return (normal[0] / length, normal[1] / length, normal[2] / length)


def _source_positions(mesh: ET.Element, source_id: str) -> List[Vector3]:
    source = mesh.find(f"c:source[@id='{source_id}']", COLLADA_NS)
    if source is None:
        raise ValueError(f"Missing source {source_id!r}")

    float_array = source.find("c:float_array", COLLADA_NS)
    accessor = source.find("c:technique_common/c:accessor", COLLADA_NS)
    if float_array is None or float_array.text is None or accessor is None:
        raise ValueError(f"Source {source_id!r} does not contain an accessor-backed float_array")

    stride = int(accessor.attrib.get("stride", "3"))
    offset = int(accessor.attrib.get("offset", "0"))
    count = int(accessor.attrib["count"])
    values = _parse_float_array(float_array.text)
    positions = []
    for index in range(count):
        base = offset + index * stride
        positions.append((values[base], values[base + 1], values[base + 2]))
    return positions


def _vertices_position_source(mesh: ET.Element, vertices_id: str) -> str:
    vertices = mesh.find(f"c:vertices[@id='{vertices_id}']", COLLADA_NS)
    if vertices is None:
        raise ValueError(f"Missing vertices {vertices_id!r}")
    input_element = vertices.find("c:input[@semantic='POSITION']", COLLADA_NS)
    if input_element is None:
        raise ValueError(f"Vertices {vertices_id!r} has no POSITION input")
    return _strip_fragment(input_element.attrib["source"])


def _geometry_transforms(root: ET.Element) -> Dict[str, Matrix4]:
    transforms: Dict[str, Matrix4] = {}
    for node in root.findall(".//c:visual_scene//c:node", COLLADA_NS):
        matrix = _parse_matrix((node.findtext("c:matrix", namespaces=COLLADA_NS) or "").strip())
        for instance in node.findall("c:instance_geometry", COLLADA_NS):
            url = instance.attrib.get("url")
            if url:
                transforms[_strip_fragment(url)] = matrix
    return transforms


def _mesh_triangles(mesh: ET.Element) -> List[Tuple[int, int, int]]:
    triangles: List[Tuple[int, int, int]] = []
    for triangle_element in mesh.findall("c:triangles", COLLADA_NS):
        inputs = triangle_element.findall("c:input", COLLADA_NS)
        vertex_input = next((item for item in inputs if item.attrib.get("semantic") == "VERTEX"), None)
        if vertex_input is None:
            raise ValueError("Triangles element has no VERTEX input")

        vertex_offset = int(vertex_input.attrib.get("offset", "0"))
        stride = max(int(item.attrib.get("offset", "0")) for item in inputs) + 1
        primitive_text = triangle_element.findtext("c:p", default="", namespaces=COLLADA_NS)
        primitive_indices = [int(item) for item in primitive_text.split()]
        if len(primitive_indices) % (stride * 3) != 0:
            raise ValueError("Triangle primitive index buffer is not divisible by stride * 3")

        for base in range(0, len(primitive_indices), stride * 3):
            vertex_ids = []
            for corner in range(3):
                vertex_ids.append(primitive_indices[base + corner * stride + vertex_offset])
            triangles.append((vertex_ids[0], vertex_ids[1], vertex_ids[2]))
    return triangles


def convert_dae_to_stl(source_path: Path, target_path: Path) -> int:
    root = ET.parse(source_path).getroot()
    transforms = _geometry_transforms(root)
    output_triangles: List[Tuple[Vector3, Vector3, Vector3]] = []

    for geometry in root.findall(".//c:library_geometries/c:geometry", COLLADA_NS):
        geometry_id = geometry.attrib["id"]
        matrix = transforms.get(geometry_id, IDENTITY_MATRIX)
        reverse_winding = _matrix_determinant_3x3(matrix) < 0.0
        mesh = geometry.find("c:mesh", COLLADA_NS)
        if mesh is None:
            continue

        triangles = _mesh_triangles(mesh)
        vertices_element = mesh.find("c:vertices", COLLADA_NS)
        if vertices_element is None:
            raise ValueError(f"Geometry {geometry_id!r} has no vertices element")
        position_source = _vertices_position_source(mesh, vertices_element.attrib["id"])
        positions = [_transform_point(point, matrix) for point in _source_positions(mesh, position_source)]

        for index_a, index_b, index_c in triangles:
            triangle = (positions[index_a], positions[index_b], positions[index_c])
            if reverse_winding:
                triangle = (triangle[0], triangle[2], triangle[1])
            output_triangles.append(triangle)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"Converted from {source_path.name}".encode("ascii")[:80].ljust(80, b"\0")
    with target_path.open("wb") as file:
        file.write(header)
        file.write(struct.pack("<I", len(output_triangles)))
        for triangle in output_triangles:
            file.write(struct.pack("<3f", *_normal(triangle)))
            for vertex in triangle:
                file.write(struct.pack("<3f", *vertex))
            file.write(struct.pack("<H", 0))
    return len(output_triangles)


def convert_meshes(source_mesh_dir: Path, target_mesh_dir: Path) -> None:
    for source_path in sorted(source_mesh_dir.glob("*.dae")):
        target_path = target_mesh_dir / f"{source_path.stem}.stl"
        triangle_count = convert_dae_to_stl(source_path, target_path)
        print(f"{source_path.name} -> {target_path.relative_to(LEGGED_GYM_ROOT_DIR)} ({triangle_count} triangles)")


def convert_urdf(source_urdf: Path, target_urdf: Path) -> None:
    target_urdf.parent.mkdir(parents=True, exist_ok=True)
    target_urdf.write_text(source_urdf.read_text().replace(".dae", ".stl"), encoding="utf-8")
    print(f"{source_urdf.name} -> {target_urdf.relative_to(LEGGED_GYM_ROOT_DIR)}")


def main() -> None:
    convert_meshes(SOURCE_ROBOT_DIR / "meshes", TARGET_ROBOT_DIR / "meshes")
    convert_urdf(SOURCE_ROBOT_DIR / "urdf" / "serial_t800.urdf", TARGET_ROBOT_DIR / "urdf" / "serial_t800_stl.urdf")


if __name__ == "__main__":
    main()
