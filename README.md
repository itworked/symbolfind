# Symbol Matching Visualization Tool

A Python-based tool for visualizing and matching vector paths using matplotlib. This tool provides interactive visualization of vector paths and allows for cluster detection and matching of similar symbols.

## Features
- Interactive visualization of vector paths
- Spatial hashing for efficient path lookup
- Cluster detection for connected paths
- Symbol matching based on path similarities
- Click-based interaction to highlight matching clusters

## Requirements
- Python 3.x
- matplotlib
- numpy

## Setup
1. Clone the repository
2. Install required packages:
```bash
pip install matplotlib numpy
```

## Usage
1. Run sym.py with pdf_file_name.pdf (single page of a pdf is best)
```bash
python sym.py drawings_page_11.pdf
```
2. Run plot.py to plot the vector output of the pdf
```bash
python plot.py
```

3. Interact with the visualization:
   - Click on paths to highlight clusters
   - Matching clusters will be highlighted in blue
   - Original selected cluster will be highlighted in red

## Configuration
Adjust settings in the `CONFIG` dictionary within `plot.py`:
- `search_radius`: Initial radius to find first path
- `connection_tolerance`: Distance tolerance for considering points connected
- `match_tolerance`: Tolerance for comparing point distances
- `angle_tolerance`: Tolerance for comparing angles
- `grid_size`: Size of spatial hash grid cells