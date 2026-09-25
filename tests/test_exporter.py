"""
Unit tests for Chaste simulation file syntax and ASCII exporters.
"""

from pathlib import Path

from imagetochaste.exporters.base import ChasteMesh, Element, Node
from imagetochaste.exporters.chaste_formatter import (
    export_centroids_json,
    export_chaste_nodes,
    export_chaste_vertex_mesh,
    format_elements_string,
    format_nodes_string,
)


def test_format_nodes_string_simple():
    """
    Validate simple .nodes syntax (as seen in thesis cell_centres.nodes).
    """
    coords = [(10.5, 20.25), (30.125, 40.75)]
    text = format_nodes_string(coords, header_style="simple", precision=6)
    lines = text.strip().split("\n")

    assert lines[0] == "2"
    assert lines[1] == "0 10.500000 20.250000 0"
    assert lines[2] == "1 30.125000 40.750000 0"


def test_format_nodes_string_extended():
    """
    Validate extended Chaste .node syntax: <num_nodes> <dim> <num_attr> <num_boundary_markers>.
    """
    nodes = [
        Node(node_id=0, x=1.0, y=2.0, is_boundary=False),
        Node(node_id=1, x=3.0, y=4.0, is_boundary=True),
    ]
    text = format_nodes_string(nodes, header_style="extended", precision=3)
    lines = text.strip().split("\n")

    assert lines[0] == "2 2 0 1"
    assert lines[1] == "0 1.000 2.000 0"
    assert lines[2] == "1 3.000 4.000 1"


def test_format_elements_string():
    """
    Validate Chaste .elements syntax for polygonal VertexMesh.
    Header: <num_elements> 0
    Body: <elem_id> <num_nodes> <n0> <n1> ... <nk>
    """
    elements = [
        Element(element_id=0, node_ids=[0, 1, 2, 3]),
        Element(element_id=1, node_ids=[2, 4, 5]),
    ]
    text = format_elements_string(elements)
    lines = text.strip().split("\n")

    assert lines[0] == "2 0"
    assert lines[1] == "0 4 0 1 2 3"
    assert lines[2] == "1 3 2 4 5"


def test_export_chaste_nodes_file(tmp_path: Path):
    """
    Test disk writing of .nodes file.
    """
    coords = [(100.0, 200.0), (150.0, 250.0)]
    out_file = tmp_path / "test_cells.nodes"
    exported_path = export_chaste_nodes(coords, out_file)

    assert exported_path.exists()
    content = exported_path.read_text()
    assert "2\n0 100.000000 200.000000 0\n" in content


def test_export_chaste_vertex_mesh_files(tmp_path: Path):
    """
    Test combined export of both .nodes and .elements files for VertexMesh.
    """
    nodes = [
        Node(node_id=0, x=0.0, y=0.0),
        Node(node_id=1, x=10.0, y=0.0),
        Node(node_id=2, x=10.0, y=10.0),
        Node(node_id=3, x=0.0, y=10.0),
    ]
    elements = [
        Element(element_id=0, node_ids=[0, 1, 2, 3])
    ]
    mesh = ChasteMesh(nodes=nodes, elements=elements)

    base_p = tmp_path / "output_mesh"
    node_f, elem_f = export_chaste_vertex_mesh(mesh, base_p)

    assert node_f.exists()
    assert elem_f.exists()
    assert node_f.name == "output_mesh.nodes"
    assert elem_f.name == "output_mesh.elements"


def test_export_centroids_json(tmp_path: Path):
    """
    Test centroid JSON export.
    """
    coords = [(12.5, 34.5), (56.7, 78.9)]
    out_json = tmp_path / "centroids.json"
    res = export_centroids_json(coords, out_json)

    assert res.exists()
    import json
    data = json.loads(res.read_text())
    assert len(data) == 2
    assert data[0]["cell_id"] == 0
    assert data[0]["x"] == 12.5
    assert data[0]["y"] == 34.5
