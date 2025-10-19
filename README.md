# Path Finding Visualizer

A Python Pygame application that visualizes pathfinding algorithms (DFS, BFS, Dijkstra, A\*).

## Installation

```bash
pip install -r requirements.txt
python pathfinder.py
```

## Building Executable

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name "PathFindingVisualizer" pathfinder.py
```

The executable will be in the `dist/` folder.

## Controls

- Mouse drag: Draw walls
- S: Set start point
- E: Set end point
- 1-4: Run algorithms
- C: Clear path
- H: Hard clear
- G: Generate maze
- F: Fullscreen

## Author

TRK Aashish
