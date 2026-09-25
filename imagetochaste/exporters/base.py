"""
Core data structures for simulation geometry and Chaste meshes.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Node:
    """
    Represents a spatial point or cell junction vertex.
    """
    node_id: int
    x: float
    y: float
    is_boundary: bool = False
    attributes: Optional[List[float]] = None

    @property
    def boundary_flag(self) -> int:
        return 1 if self.is_boundary else 0


@dataclass
class Element:
    """
    Represents a polygonal cell body composed of ordered junction node IDs.
    """
    element_id: int
    node_ids: List[int]
    attributes: Optional[List[float]] = None

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)


@dataclass
class ChasteMesh:
    """
    Complete geometric mesh consisting of vertices (nodes) and polygonal cells (elements).
    """
    nodes: List[Node] = field(default_factory=list)
    elements: List[Element] = field(default_factory=list)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_elements(self) -> int:
        return len(self.elements)
