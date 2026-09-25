"""
Chaste C++ simulation engine formatters and file writers.

Generates valid .nodes and .elements files compatible with:
- NodesOnlyMesh (NodeBasedCellPopulation)
- VertexMeshReader (VertexBasedCellPopulation)
- TrianglesMeshReader (MeshBasedCellPopulation)
"""

import json
from io import StringIO
from pathlib import Path
from typing import List, Tuple, Union

from imagetochaste.exporters.base import ChasteMesh, Element, Node


def format_nodes_string(
    nodes: List[Union[Node, Tuple[float, float]]],
    header_style: str = "simple",
    precision: int = 6,
) -> str:
    """
    Format a list of nodes or 2D coordinates into Chaste .nodes ASCII text.

    Args:
        nodes: List of Node objects or (x, y) coordinate tuples.
        header_style:
            - 'simple': Single integer count on first line (e.g. '208').
            - 'extended': 'num_nodes num_dims num_attributes num_boundary_markers' (e.g. '208 2 0 1').
        precision: Decimal precision for floating point coordinates.

    Returns:
        Formatted ASCII text string.
    """
    n_count = len(nodes)
    buf = StringIO()

    if header_style == "extended":
        buf.write(f"{n_count} 2 0 1\n")
    else:
        buf.write(f"{n_count}\n")

    coord_fmt = f"{{:.{precision}f}}"

    for idx, item in enumerate(nodes):
        if isinstance(item, Node):
            nid = item.node_id
            x = coord_fmt.format(item.x)
            y = coord_fmt.format(item.y)
            bflag = item.boundary_flag
        else:
            nid = idx
            x = coord_fmt.format(float(item[0]))
            y = coord_fmt.format(float(item[1]))
            bflag = 0

        buf.write(f"{nid} {x} {y} {bflag}\n")

    return buf.getvalue()


def format_elements_string(
    elements: List[Element],
) -> str:
    """
    Format a list of polygonal cell elements into Chaste .elements ASCII text.

    Chaste VertexMesh element format:
    Header: <num_elements> <num_attributes>
    Body:   <element_id> <num_nodes> <n0> <n1> ... <nk> [<attributes>]

    Args:
        elements: List of Element objects with counter-clockwise node ID sequences.

    Returns:
        Formatted ASCII text string.
    """
    n_elems = len(elements)
    buf = StringIO()
    buf.write(f"{n_elems} 0\n")

    for elem in elements:
        node_str = " ".join(str(nid) for nid in elem.node_ids)
        buf.write(f"{elem.element_id} {elem.num_nodes} {node_str}\n")

    return buf.getvalue()


def export_chaste_nodes(
    nodes_or_centroids: List[Union[Node, Tuple[float, float]]],
    output_path: Union[str, Path],
    header_style: str = "simple",
    precision: int = 6,
    create_chaste_native_alias: bool = True,
) -> Path:
    """
    Export nodes or cell centroids to a Chaste-compatible .nodes ASCII file.

    Args:
        nodes_or_centroids: List of Node objects or (x, y) tuples.
        output_path: Destination path for .nodes file.
        header_style: 'simple' or 'extended'.
        precision: Float coordinate precision.
        create_chaste_native_alias: Whether to also write .node (or .nodes) alias.

    Returns:
        Path to written file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    content = format_nodes_string(nodes_or_centroids, header_style=header_style, precision=precision)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(content)

    if create_chaste_native_alias:
        if out_file.suffix == ".nodes":
            with open(out_file.with_suffix(".node"), "w", encoding="utf-8") as f:
                f.write(content)
        elif out_file.suffix == ".node":
            with open(out_file.with_suffix(".nodes"), "w", encoding="utf-8") as f:
                f.write(content)

    return out_file


def export_chaste_vertex_mesh(
    mesh: ChasteMesh,
    base_output_path: Union[str, Path],
    nodes_header_style: str = "extended",
    precision: int = 6,
    write_chaste_native_extensions: bool = True,
) -> Tuple[Path, Path]:
    """
    Export a complete Chaste VertexMesh (both .nodes and .elements files).

    Also writes native Chaste C++ files (.node and .cell) so that Oxford Chaste's
    VertexMeshReader<2, 2> can load the mesh directly from the base path.

    Given base path '/path/to/cell_mesh', writes:
      - /path/to/cell_mesh.nodes and /path/to/cell_mesh.node
      - /path/to/cell_mesh.elements and /path/to/cell_mesh.cell

    Args:
        mesh: ChasteMesh instance containing nodes and elements.
        base_output_path: Base path without extension or path ending in .nodes / .elements.
        nodes_header_style: 'extended' or 'simple'.
        precision: Coordinate float precision.
        write_chaste_native_extensions: Whether to also generate native .node and .cell files.

    Returns:
        Tuple of (nodes_file_path, elements_file_path).
    """
    base = Path(base_output_path)
    if base.suffix in [".nodes", ".node", ".elements", ".element", ".cell", ".ele", ".txt"]:
        base_stem = base.with_suffix("")
    else:
        base_stem = base

    base_stem.parent.mkdir(parents=True, exist_ok=True)
    nodes_file = base_stem.with_suffix(".nodes")
    elements_file = base_stem.with_suffix(".elements")

    # Write nodes
    nodes_content = format_nodes_string(mesh.nodes, header_style=nodes_header_style, precision=precision)
    with open(nodes_file, "w", encoding="utf-8") as f:
        f.write(nodes_content)

    # Write elements
    elements_content = format_elements_string(mesh.elements)
    with open(elements_file, "w", encoding="utf-8") as f:
        f.write(elements_content)

    # Also write native Chaste C++ extensions (.node and .cell) for VertexMeshReader
    if write_chaste_native_extensions:
        with open(base_stem.with_suffix(".node"), "w", encoding="utf-8") as f:
            f.write(nodes_content)
        with open(base_stem.with_suffix(".cell"), "w", encoding="utf-8") as f:
            f.write(elements_content)

    return nodes_file, elements_file


def export_centroids_json(
    centroids: List[Tuple[float, float]],
    output_path: Union[str, Path],
) -> Path:
    """
    Export cell centroids to a structured JSON file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    data = [{"cell_id": idx, "x": float(x), "y": float(y)} for idx, (x, y) in enumerate(centroids)]
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return out_file
