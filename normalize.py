import json
import numpy as np
import os

def load_symbol(symbol_path):
    with open(symbol_path, 'r') as f:
        symbol = json.load(f)
    return symbol

def save_symbol(symbol, symbol_path):
    with open(symbol_path, 'w') as f:
        json.dump(symbol, f, indent=4)

def normalize_symbol(symbol):
    """
    Normalizes the vector paths of a symbol.
    Updates the 'normalized_vector_paths' field in the symbol data structure.
    """
    all_points = []
    commands_list = []
    
    # Collect all points from vector paths
    for path in symbol['vector_paths']:
        commands = path['commands']
        for cmd in commands:
            points = cmd['points']
            all_points.extend(points)
        commands_list.append(commands)
    
    # Convert to numpy array for calculations
    all_points = np.array(all_points)
    
    # Calculate centroid
    centroid = np.mean(all_points, axis=0)
    
    # Center the points
    centered_points = all_points - centroid
    
    # Calculate scale (e.g., maximum distance from centroid)
    scale = np.max(np.linalg.norm(centered_points, axis=1))
    
    # Avoid division by zero
    if scale == 0:
        scale = 1.0
    
    # Normalize scale
    normalized_points = centered_points / scale
    
    # Optionally, rotate to align with principal axis
    # Calculate covariance matrix
    cov_matrix = np.cov(normalized_points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # Sort eigenvectors by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    # Rotate points
    rotated_points = normalized_points @ eigenvectors
    
    # Update normalized vector paths
    idx = 0
    normalized_vector_paths = []
    for path_commands in commands_list:
        normalized_commands = []
        for cmd in path_commands:
            num_points = len(cmd['points'])
            cmd_points = rotated_points[idx:idx+num_points].tolist()
            idx += num_points
            normalized_commands.append({
                'command': cmd['command'],
                'points': cmd_points
            })
        normalized_vector_paths.append({
            'stroke_color': path.get('stroke_color'),
            'fill_color': path.get('fill_color'),
            'line_width': path.get('line_width'),
            'commands': normalized_commands
        })
    
    # Update the symbol data structure
    symbol['normalized_vector_paths'] = normalized_vector_paths
    
    return symbol

def main():
    # Directory containing raw symbol JSON files
    symbols_dir = 'symbols_raw'
    # Directory to save normalized symbol JSON files
    normalized_dir = 'symbols_normalized'
    os.makedirs(normalized_dir, exist_ok=True)
    
    # Process each symbol file
    for filename in os.listdir(symbols_dir):
        if filename.endswith('.json'):
            symbol_path = os.path.join(symbols_dir, filename)
            symbol = load_symbol(symbol_path)
            # Normalize the symbol
            symbol = normalize_symbol(symbol)
            # Save the normalized symbol
            normalized_path = os.path.join(normalized_dir, filename)
            save_symbol(symbol, normalized_path)
            print(f"Normalized and saved symbol: {filename}")

if __name__ == "__main__":
    main()
