import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
from collections import defaultdict
import time

CONFIG = {
    'search_radius': 50,      # Initial radius to find first path
    'connection_tolerance': 1.0,  # Distance tolerance for considering points connected
    'match_tolerance': 2.0,    # Tolerance for comparing point distances in matching symbols
    'angle_tolerance': 0.2,    # Tolerance for comparing angles (in radians)
    'grid_size': 2.0,         # Size of spatial hash grid cells (> connection_tolerance)
}

class SpatialHash:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.grid = defaultdict(set)
        self.path_cells = defaultdict(set)  # Track which cells each path is in
        
    def get_cell(self, point):
        """Convert a point to grid cell coordinates"""
        x, y = point
        return (int(x / self.grid_size), int(y / self.grid_size))
    
    def add_path(self, path_id, points):
        """Add a path's points to the spatial hash"""
        for point in points:
            cell = self.get_cell(point)
            self.grid[cell].add(path_id)
            self.path_cells[path_id].add(cell)
    
    def get_nearby_paths(self, points):
        """Get all paths that could potentially connect to these points"""
        nearby_paths = set()
        cells_checked = set()
        
        for point in points:
            base_cell = self.get_cell(point)
            # Check surrounding cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    cell = (base_cell[0] + dx, base_cell[1] + dy)
                    if cell not in cells_checked:
                        cells_checked.add(cell)
                        nearby_paths.update(self.grid[cell])
        
        return nearby_paths

class SymbolMatcher:
    def __init__(self, vector_paths):
        self.vector_paths = vector_paths
        self.path_dict = {id(path): path for path in vector_paths}
        self.spatial_hash = SpatialHash(CONFIG['grid_size'])
        self.clusters = []
        
        # UI setup
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.current_highlights = []
        self.toolbar = plt.get_current_fig_manager().toolbar
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Initialize spatial hash and find clusters
        self.initialize_spatial_hash()
        self.find_all_clusters()
        
        # Plot paths
        self.plot_all_paths()

    def get_path_points(self, path):
        """Extract all points from a path's commands"""
        points = []
        current_pos = None
        for cmd in path['commands']:
            if cmd['command'] == 'm':
                current_pos = tuple(cmd['points'][0])
                points.append(current_pos)
            elif cmd['command'] == 'l' and current_pos:
                for point in cmd['points']:
                    end_pos = tuple(point)
                    if current_pos != end_pos:
                        points.append(end_pos)
                    current_pos = end_pos
            elif cmd['command'] == 'c' and current_pos and len(cmd['points']) == 3:
                points.extend([tuple(p) for p in cmd['points']])
                current_pos = tuple(cmd['points'][-1])
        return points

    def initialize_spatial_hash(self):
        """Build spatial hash for all paths"""
        start_time = time.time()
        print("\nBuilding spatial hash...")
        point_count = 0
        
        for path in self.vector_paths:
            points = self.get_path_points(path)
            point_count += len(points)
            self.spatial_hash.add_path(id(path), points)
        
        duration = time.time() - start_time
        print(f"Added {len(self.vector_paths)} paths with {point_count} points to spatial hash in {duration:.2f} seconds")
        
        # Print grid statistics
        cell_count = len(self.spatial_hash.grid)
        paths_per_cell = [len(paths) for paths in self.spatial_hash.grid.values()]
        avg_paths_per_cell = sum(paths_per_cell) / cell_count if cell_count > 0 else 0
        max_paths_per_cell = max(paths_per_cell) if paths_per_cell else 0
        
        print(f"Grid statistics:")
        print(f"- Total cells: {cell_count}")
        print(f"- Average paths per cell: {avg_paths_per_cell:.1f}")
        print(f"- Maximum paths per cell: {max_paths_per_cell}")

    def are_paths_connected(self, path1, path2):
        """Check if two paths are connected within tolerance"""
        points1 = self.get_path_points(path1)
        points2 = self.get_path_points(path2)
        
        for p1 in points1:
            p1_array = np.array(p1)
            for p2 in points2:
                p2_array = np.array(p2)
                if np.linalg.norm(p1_array - p2_array) <= CONFIG['connection_tolerance']:
                    return True
        return False

    def find_cluster(self, start_path):
        """Find all paths connected to the start path using spatial hash"""
        cluster = {id(start_path): start_path}
        to_check = {id(start_path)}
        checked = set()
        
        while to_check:
            current_path_id = to_check.pop()
            current_path = self.path_dict[current_path_id]
            checked.add(current_path_id)
            
            # Get points for current path
            current_points = self.get_path_points(current_path)
            
            # Get potentially connected paths from spatial hash
            nearby_path_ids = self.spatial_hash.get_nearby_paths(current_points)
            
            # Check each nearby path
            for path_id in nearby_path_ids:
                if path_id not in checked and path_id not in to_check:
                    path = self.path_dict[path_id]
                    if self.are_paths_connected(current_path, path):
                        cluster[path_id] = path
                        to_check.add(path_id)
        
        return list(cluster.values())

    def find_all_clusters(self):
        """Find all clusters in the drawing"""
        start_time = time.time()
        print("\nFinding all clusters...")
        
        processed_paths = set()
        comparison_count = 0
        
        for path in self.vector_paths:
            path_id = id(path)
            if path_id not in processed_paths:
                cluster = self.find_cluster(path)
                for p in cluster:
                    processed_paths.add(id(p))
                self.clusters.append(cluster)
                comparison_count += len(cluster)
        
        duration = time.time() - start_time
        print(f"Found {len(self.clusters)} clusters in {duration:.2f} seconds")
        print(f"Average cluster size: {sum(len(c) for c in self.clusters) / len(self.clusters):.1f} paths")
        print(f"Total path-to-path comparisons: {comparison_count}")

    def get_nearest_path(self, x, y):
        """Find the path closest to the click point"""
        click_point = np.array([x, y])
        nearest_path = None
        min_distance = float('inf')
        
        # Use spatial hash to only check nearby paths
        nearby_paths = self.spatial_hash.get_nearby_paths([(x, y)])
        
        for path_id in nearby_paths:
            path = self.path_dict[path_id]
            for cmd in path['commands']:
                points = cmd['points']
                for point in points:
                    if isinstance(point, (list, tuple)) and len(point) == 2:
                        dist = np.linalg.norm(np.array(point) - click_point)
                        if dist < min_distance:
                            min_distance = dist
                            nearest_path = path
        
        return nearest_path, min_distance

    def on_click(self, event):
        if self.toolbar.mode != '' or event.inaxes != self.ax:
            return
            
        nearest_path, distance = self.get_nearest_path(event.xdata, event.ydata)
        if nearest_path is None:
            return
            
        print(f"\nNearest path is {distance:.1f} units away")
        
        # Find which cluster contains this path
        clicked_cluster = None
        for cluster in self.clusters:
            if any(id(path) == id(nearest_path) for path in cluster):
                clicked_cluster = cluster
                break
        
        if clicked_cluster:
            print(f"Found cluster with {len(clicked_cluster)} paths")
            matches = self.find_matching_clusters(clicked_cluster)
            # Update all highlights at once
            self.update_highlights([clicked_cluster] + matches)
        
    def plot_all_paths(self):
        """Plot all vector paths"""
        self.path_elements = []
        all_x, all_y = [], []
        
        for path in self.vector_paths:
            elements = self.plot_single_path(path)
            self.path_elements.extend(elements)
            
            # Collect bounds
            for elem in elements:
                if hasattr(elem, 'get_path'):
                    vertices = elem.get_path().vertices
                    all_x.extend(vertices[:, 0])
                    all_y.extend(vertices[:, 1])
                elif hasattr(elem, 'get_data'):
                    x_data, y_data = elem.get_data()
                    all_x.extend(x_data)
                    all_y.extend(y_data)
        
        if all_x and all_y:
            margin_x = (max(all_x) - min(all_x)) * 0.05
            margin_y = (max(all_y) - min(all_y)) * 0.05
            plt.xlim(min(all_x) - margin_x, max(all_x) + margin_x)
            plt.ylim(max(all_y) + margin_y, min(all_y) - margin_y)
            
        plt.axis('equal')
        plt.tight_layout()
    
    def plot_single_path(self, path):
        elements = []
        commands = path['commands']
        current_pos = None
        start_pos = None
        
        for cmd in commands:
            command = cmd['command']
            points = cmd['points']
            
            if command == 'm':
                current_pos = tuple(points[0])
                start_pos = current_pos
            elif command == 'l':
                if current_pos is None:
                    continue
                for point in points:
                    point = tuple(point)
                    if current_pos != point:
                        line = plt.plot([current_pos[0], point[0]], 
                                      [current_pos[1], point[1]],
                                      linewidth=path.get('line_width', 1.0),
                                      color='black')[0]
                        elements.append(line)
                    current_pos = point
            elif command == 'c':
                if current_pos is None or len(points) != 3:
                    continue
                cp1, cp2, end_pt = [tuple(p) for p in points]
                verts = [current_pos, cp1, cp2, end_pt]
                codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                path_patch = patches.PathPatch(
                    Path(verts, codes),
                    linewidth=path.get('line_width', 1.0),
                    edgecolor='black',
                    facecolor='none'
                )
                self.ax.add_patch(path_patch)
                elements.append(path_patch)
                current_pos = end_pt
        
        return elements
    
    def highlight_paths(self, paths, color='red'):
        """Highlight a set of paths with the given color"""
        elements = []
        for path in paths:
            path_elements = self.plot_single_path(path)
            for element in path_elements:
                if isinstance(element, patches.PathPatch):
                    highlight = patches.PathPatch(
                        element.get_path(),
                        linewidth=element.get_linewidth() + 1,
                        edgecolor=color,
                        facecolor='none',
                        zorder=10
                    )
                    self.ax.add_patch(highlight)
                    elements.append(highlight)
                elif hasattr(element, 'get_data'):
                    x_data, y_data = element.get_data()
                    highlight = plt.plot(
                        x_data, y_data,
                        linewidth=element.get_linewidth() + 1,
                        color=color,
                        zorder=10
                    )[0]
                    elements.append(highlight)
        return elements

    def update_highlights(self, clusters):
        """Update all highlights at once"""
        # Clear previous highlights
        for highlight in self.current_highlights:
            highlight.remove()
        self.current_highlights = []
        
        # Highlight clicked cluster in red
        self.current_highlights.extend(self.highlight_paths(clusters[0], 'red'))
        
        # Highlight matches in blue
        for cluster in clusters[1:]:
            self.current_highlights.extend(self.highlight_paths(cluster, 'blue'))
        
        self.fig.canvas.draw_idle()

    def get_cluster_fingerprint(self, cluster):
        """Get normalized geometric representation of the cluster"""
        # Get all points
        all_points = [point for path in cluster 
                    for point in self.get_path_points(path)]
        if not all_points:
            return None
            
        # Convert to numpy array for efficient computation
        points = np.array(all_points)
        
        # Calculate center and normalize positions
        center = np.mean(points, axis=0)
        centered = points - center
        
        # Scale to make size-invariant
        max_dist = np.max(np.linalg.norm(centered, axis=1))
        if max_dist == 0:
            return None
        normalized = centered / max_dist
        
        # Sort points by angle and distance from center for consistent comparison
        angles = np.arctan2(normalized[:, 1], normalized[:, 0])
        distances = np.linalg.norm(normalized, axis=1)
        sorted_indices = np.lexsort((distances, angles))
        
        return {
            'path_count': len(cluster),
            'points': normalized[sorted_indices].tolist()
        }

    def clusters_match(self, fp1, fp2):
        """Compare two cluster fingerprints"""
        if fp1 is None or fp2 is None:
            return False
            
        # Quick path count check
        if abs(fp1['path_count'] - fp2['path_count']) > 1:
            return False
        
        points1 = np.array(fp1['points'])
        points2 = np.array(fp2['points'])
        
        # Allow for some point count variation
        if abs(len(points1) - len(points2)) > max(2, min(len(points1), len(points2)) * 0.1):
            return False
        
        # If point counts differ, use the smaller set
        n_points = min(len(points1), len(points2))
        points1 = points1[:n_points]
        points2 = points2[:n_points]
        
        # Calculate point-wise distances
        distances = np.linalg.norm(points1 - points2, axis=1)
        
        # Use a relative tolerance based on cluster size
        tolerance = CONFIG['match_tolerance'] / 100  # Normalized space
        
        # Allow some points to be off while still considering it a match
        max_mismatched_points = max(1, n_points // 10)  # 10% of points can be off
        return np.sum(distances > tolerance) <= max_mismatched_points

    def find_matching_clusters(self, target_cluster):
        """Find clusters that match the target cluster"""
        target_fp = self.get_cluster_fingerprint(target_cluster)
        if target_fp is None:
            return []
        
        # First pass - filter by path count
        candidates = [c for c in self.clusters 
                    if abs(len(c) - target_fp['path_count']) <= 1 and
                    c[0] not in target_cluster]  # Avoid self-match
        
        print(f"\nChecking {len(candidates)} candidates with similar path counts...")
        
        # Detailed geometric matching
        matches = []
        for cluster in candidates:
            fp = self.get_cluster_fingerprint(cluster)
            if self.clusters_match(target_fp, fp):
                matches.append(cluster)
        
        print(f"Found {len(matches)} matching clusters")
        return matches

# Load and run
with open('vector_paths.json', 'r') as f:
    vector_paths = json.load(f)

print(f"Loaded {len(vector_paths)} vector paths")
print(f"\nCurrent settings:")
for key, value in CONFIG.items():
    print(f"{key}: {value}")

matcher = SymbolMatcher(vector_paths)
plt.show()