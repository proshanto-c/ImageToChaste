# Chaste Simulation Compatibility & Integration Guide

`ImageToChaste` produces standard ASCII mesh files directly compatible with the [Chaste](https://chaste.github.io/) (Cancer, Heart and Soft Tissue Environment) C++ simulation framework developed at the University of Oxford.

---

## 1. Supported Simulation Paradigms

| Simulation Model | Chaste Class | Required Files | ImageToChaste CLI Mode |
|---|---|---|---|
| **Center-Based / Off-Lattice** | `NodeBasedCellPopulation<2>` | `*.nodes` | `--mode centroids` |
| **Vertex Model** | `VertexBasedCellPopulation<2>` | `*.nodes`, `*.elements` | `--mode voronoi` |
| **Delaunay / Mesh-Based** | `MeshBasedCellPopulation<2>` | `*.nodes` (or Triangles format) | `--mode centroids` |

---

## 2. File Syntax & Extension Specifications

In Oxford Chaste C++, mesh readers take a **base file path** and automatically append native extensions:
- **`VertexMeshReader<2, 2>("path/to/mesh")`** opens `path/to/mesh.node` and `path/to/mesh.cell`.
- **`TrianglesMeshReader<2, 2>("path/to/mesh")`** opens `path/to/mesh.node` and `path/to/mesh.ele`.

In computational biology literature and thesis exploratory scripts, researchers frequently name these `.nodes` and `.elements`. To provide seamless compatibility for both Chaste C++ binaries and research workflows, `ImageToChaste` **automatically outputs both pairs** (`.node` / `.cell` and `.nodes` / `.elements`).

### 2.1 Nodes File (`*.node` / `*.nodes`)
Header and line syntax parsed by Chaste:

```text
<num_nodes> [optional: <num_dims> <num_attributes> <num_boundary_markers>]
<node_id> <x_coord> <y_coord> <is_boundary_marker>
```

#### Example Output:
```text
208 2 0 1
0 364.765625 124.750000 0
1 330.715013 140.430025 0
2 328.043478 171.568116 0
...
```

- Coordinate units match pixel space (or physical microns if scaled).
- `is_boundary_marker` is `0` for internal cell nodes and `1` for peripheral tissue boundary nodes.

---

### 2.2 Elements / Cells File (`*.cell` / `*.elements`)
Header and line syntax parsed by Chaste `VertexMeshReader`:

```text
<num_elements> <num_attributes>
<element_id> <num_nodes> <node_0> <node_1> ... <node_k>
```

#### Example Output:
```text
25 0
0 6 12 15 18 19 23 24
1 5 18 20 21 22 25
...
```

- Each line represents a single cell polygon.
- Node IDs must be ordered **counter-clockwise** (CCW) to maintain positive polygon area and correct outward normal vectors in Chaste mechanical force calculations.

---

## 3. C++ Chaste Integration Example

Below is a complete C++ test script demonstrating how to ingest `ImageToChaste` outputs into a 2D Chaste simulation:

```cpp
#include <cxxtest/TestSuite.h>
#include "CellBasedSimulationArchiver.hpp"
#include "OffLatticeSimulation.hpp"
#include "NodesOnlyMesh.hpp"
#include "NodeBasedCellPopulation.hpp"
#include "VertexMeshReader.hpp"
#include "MutableVertexMesh.hpp"
#include "VertexBasedCellPopulation.hpp"
#include "DifferentiatedCellProliferativeType.hpp"
#include "NagaiHondaDifferentialAdhesionForce.hpp"
#include "SimpleTargetAreaModifier.hpp"

class TestImageToChasteSimulation : public CxxTest::TestSuite
{
public:
    void TestVertexModelFromImageMesh()
    {
        // 1. Read files exported by ImageToChaste
        VertexMeshReader<2, 2> mesh_reader("data/sample/chaste_mesh");
        MutableVertexMesh<2, 2> cell_mesh;
        cell_mesh.ConstructFromMeshReader(mesh_reader);

        // 2. Initialise cell population
        std::vector<CellPtr> cells;
        MAKE_PTR(DifferentiatedCellProliferativeType, p_diff_type);
        CellsGenerator<FixedG1GenerationalCellCycleModel, 2> cells_generator;
        cells_generator.GenerateBasic(cells, cell_mesh.GetNumElements(), std::vector<unsigned>(), p_diff_type);

        VertexBasedCellPopulation<2> cell_population(cell_mesh, cells);

        // 3. Configure Off-Lattice Simulation
        OffLatticeSimulation<2> simulator(cell_population);
        simulator.SetOutputDirectory("SimulationFromMicroscopyScan");
        simulator.SetEndTime(10.0);
        simulator.SetDt(0.01);

        // 4. Attach forces and growth modifiers
        MAKE_PTR(NagaiHondaDifferentialAdhesionForce<2>, p_force);
        simulator.AddForce(p_force);

        MAKE_PTR(SimpleTargetAreaModifier<2>, p_growth_modifier);
        simulator.AddSimulationModifier(p_growth_modifier);

        // 5. Run simulation
        simulator.Solve();
    }
};
```
