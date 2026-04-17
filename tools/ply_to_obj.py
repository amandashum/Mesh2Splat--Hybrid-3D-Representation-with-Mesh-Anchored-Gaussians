from __future__ import annotations

import argparse
import io
import struct
from pathlib import Path


PLY_TO_STRUCT = {
    "char": "b",
    "uchar": "B",
    "short": "h",
    "ushort": "H",
    "int": "i",
    "uint": "I",
    "float": "f",
    "double": "d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a triangle mesh PLY file into OBJ.")
    parser.add_argument("--input", required=True, help="Path to an ASCII or binary_little_endian PLY file.")
    parser.add_argument("--output", required=True, help="Path to the OBJ file to write.")
    return parser.parse_args()


def _parse_header(handle) -> tuple[str, list[str], list[str], tuple[str, str] | None, int, int]:
    header_lines: list[str] = []
    while True:
        line = handle.readline()
        if not line:
            raise ValueError("PLY file ended before end_header.")
        stripped = line.decode("ascii").strip()
        header_lines.append(stripped)
        if stripped == "end_header":
            break

    if not header_lines or header_lines[0] != "ply":
        raise ValueError("Input is not a PLY file.")

    format_line = next((line for line in header_lines if line.startswith("format ")), None)
    if format_line is None:
        raise ValueError("PLY header did not declare a format.")
    file_format = format_line.split()[1]

    vertex_count = None
    face_count = None
    vertex_properties: list[str] = []
    face_list_property: tuple[str, str] | None = None
    current_element = None

    for line in header_lines:
        parts = line.split()
        if not parts:
            continue
        if len(parts) == 3 and parts[0] == "element":
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
            elif current_element == "face":
                face_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) >= 3:
                vertex_properties.append(parts[1])
        elif parts[0] == "property" and current_element == "face":
            if len(parts) == 5 and parts[1] == "list" and parts[4] == "vertex_indices":
                face_list_property = (parts[2], parts[3])

    if vertex_count is None or face_count is None:
        raise ValueError("PLY header is missing vertex or face counts.")
    if face_list_property is None:
        raise ValueError("PLY header is missing a face vertex_indices list property.")

    return file_format, vertex_properties, header_lines, face_list_property, vertex_count, face_count


def _triangulate(indices: list[int]) -> list[tuple[int, int, int]]:
    triangles: list[tuple[int, int, int]] = []
    for face_index in range(1, len(indices) - 1):
        triangles.append((indices[0], indices[face_index], indices[face_index + 1]))
    return triangles


def _binary_reader(dtype: str):
    if dtype not in PLY_TO_STRUCT:
        raise ValueError(f"Unsupported PLY data type '{dtype}'.")
    code = PLY_TO_STRUCT[dtype]
    size = struct.calcsize("<" + code)

    def read_value(handle):
        raw = handle.read(size)
        if len(raw) != size:
            raise ValueError("PLY file ended unexpectedly while reading binary data.")
        return struct.unpack("<" + code, raw)[0]

    return read_value


def load_ply(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    # COLMAP meshes often contain extra vertex attributes and may store face
    # list counts as 32-bit integers. Parse the header-driven layout instead of
    # assuming a hard-coded PLY shape.
    with path.open("rb") as handle:
        file_format, vertex_properties, _, face_list_property, vertex_count, face_count = _parse_header(handle)
        face_count_type, face_index_type = face_list_property

        if file_format == "ascii":
            text_handle = io.TextIOWrapper(handle, encoding="utf-8")
            vertices: list[tuple[float, float, float]] = []
            for _ in range(vertex_count):
                parts = text_handle.readline().strip().split()
                if len(parts) < len(vertex_properties):
                    raise ValueError(f"{path} contains an invalid vertex row.")
                vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))

            faces: list[tuple[int, int, int]] = []
            for _ in range(face_count):
                parts = text_handle.readline().strip().split()
                if not parts:
                    raise ValueError(f"{path} contains an invalid face row.")
                vertex_per_face = int(parts[0])
                if len(parts) < 1 + vertex_per_face:
                    raise ValueError(f"{path} ended while reading face indices.")
                indices = [int(value) for value in parts[1 : 1 + vertex_per_face]]
                if vertex_per_face >= 3:
                    faces.extend(_triangulate(indices))
            return vertices, faces

        if file_format != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format '{file_format}'.")

        vertex_readers = [_binary_reader(dtype) for dtype in vertex_properties]
        read_face_count = _binary_reader(face_count_type)
        read_face_index = _binary_reader(face_index_type)

        vertices: list[tuple[float, float, float]] = []
        for _ in range(vertex_count):
            values = [reader(handle) for reader in vertex_readers]
            if len(values) < 3:
                raise ValueError(f"{path} does not expose x/y/z as the first vertex properties.")
            vertices.append((float(values[0]), float(values[1]), float(values[2])))

        faces: list[tuple[int, int, int]] = []
        for _ in range(face_count):
            vertex_per_face = int(read_face_count(handle))
            indices = [int(read_face_index(handle)) for _ in range(vertex_per_face)]
            if vertex_per_face >= 3:
                faces.extend(_triangulate(indices))

    return vertices, faces


def write_obj(path: Path, vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Generated from PLY by tools/ply_to_obj.py\n")
        for x, y, z in vertices:
            handle.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
        for i, j, k in faces:
            handle.write(f"f {i + 1} {j + 1} {k + 1}\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    vertices, faces = load_ply(input_path)
    if not vertices or not faces:
        raise ValueError(f"{input_path} did not contain any vertices/faces to write.")
    write_obj(output_path, vertices, faces)
    print(f"Wrote OBJ mesh to: {output_path}")


if __name__ == "__main__":
    main()
