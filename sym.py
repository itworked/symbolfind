import fitz  # PyMuPDF
import json
import argparse
from collections import defaultdict

def serialize_point(pt):
    """Converts a fitz.Point or list/tuple to a tuple of (x, y)."""
    if isinstance(pt, fitz.Point):
        return (pt.x, pt.y)
    elif isinstance(pt, (list, tuple)) and len(pt) == 2:
        return tuple(map(float, pt))
    elif isinstance(pt, fitz.Rect):
        return (float(pt.x0), float(pt.y0))
    else:
        return None

def is_visible_element(drawing):
    """Determine if a drawing element should be visible in the output."""
    # Check for rectangles that are likely text backgrounds
    if len(drawing['items']) == 1 and drawing['items'][0][0] == 're':
        # If it's a filled rectangle with no stroke, it's probably a text background
        has_fill = drawing.get('fill') is not None
        has_stroke = drawing.get('stroke') is not None
        if has_fill and not has_stroke:
            return False
    return True

def handle_line(item):
    """Extract start and end points from a line command."""
    if len(item) >= 2:
        start_pt = serialize_point(item[1])
        end_pt = serialize_point(item[2]) if len(item) > 2 else None
        return start_pt, end_pt
    return None, None

def handle_rect(item):
    """Extract rectangle dimensions from a rect command."""
    if len(item) >= 2 and isinstance(item[1], fitz.Rect):
        rect = item[1]
        return [rect.x0, rect.y0, rect.width, rect.height]
    return None
def process_curve_points(items, start_idx):
    """Process all curve points that form a complete circle."""
    if start_idx + 3 >= len(items):  # Need at least 4 curves for a circle
        return None, 0

    # Check if we have 4 consecutive 'c' commands
    if not all(items[start_idx + i][0] == 'c' for i in range(4)):
        return None, 0

    # Create path starting with move to first point
    path_points = []
    first_point = serialize_point(items[start_idx][1])
    if not first_point:
        return None, 0
    path_points.append(first_point)

    # Process all 4 curves
    for i in range(4):
        item = items[start_idx + i]
        if len(item) < 4:  # Need start, control1, control2, end for each curve
            return None, 0
            
        # In PDF curves, all points are meaningful:
        # item[1] is current point (already handled as start or previous endpoint)
        # item[2] is first control point
        # item[3] is second control point/endpoint
        control1 = serialize_point(item[2])
        control2 = serialize_point(item[3])
        
        # Get endpoint from next curve's start point, or first point to close
        next_idx = start_idx + (i + 1) % 4
        end_point = serialize_point(items[next_idx][1])
        
        if not all([control1, control2, end_point]):
            return None, 0
            
        path_points.extend([control1, control2, end_point])

    return path_points, 4

def main():
    parser = argparse.ArgumentParser(description="Extract vector graphics and text from a PDF.")
    parser.add_argument('pdf_file', type=str, help="The PDF file to process")
    parser.add_argument('--page', type=int, default=0, help="Page number to extract")
    parser.add_argument('--debug', action='store_true', help="Enable debug output")
    args = parser.parse_args()

    doc = fitz.open(args.pdf_file)
    page = doc[args.page]
    drawings = page.get_drawings()

    print(f"Page {args.page} content summary:")
    print(f"  Vector graphics (drawings): {len(drawings)}")

    vector_paths = []
    command_counts = defaultdict(int)

    for drawing in drawings:
        if not drawing['items'] or not is_visible_element(drawing):
            continue

        items = drawing['items']
        first_item = items[0]
        command = first_item[0]

        if args.debug and command == 'c':
            print("Processing curve set...")
            curve_points, consumed = process_curve_points(items, 0)
            if curve_points:
                print(f"Successfully processed curve with {len(curve_points)} points")
            else:
                print("Failed to process curve")

        path = {
            'stroke_color': drawing.get('stroke'),
            'fill_color': drawing.get('fill'),
            'line_width': drawing.get('width'),
            'commands': []
        }

        if command == 're':
            rect_dims = handle_rect(first_item)
            if rect_dims:
                path['commands'].append({
                    'command': 're',
                    'points': [rect_dims]
                })
                command_counts['re'] += 1
                vector_paths.append(path)
            continue

        if command == 'l':
            start_pt, end_pt = handle_line(first_item)
            if start_pt and end_pt:
                path['commands'].append({
                    'command': 'm',
                    'points': [start_pt]
                })
                command_counts['m'] += 1
                
                path['commands'].append({
                    'command': 'l',
                    'points': [end_pt]
                })
                command_counts['l'] += 1
                vector_paths.append(path)
            continue

        if command == 'c':
            curve_points, consumed = process_curve_points(items, 0)
            if curve_points:
                path['commands'].append({
                    'command': 'm',
                    'points': [curve_points[0]]  # Start point
                })
                command_counts['m'] += 1
                
                # Add each curve segment with its control points
                for i in range(1, len(curve_points), 3):
                    path['commands'].append({
                        'command': 'c',
                        'points': curve_points[i:i+3]
                    })
                    command_counts['c'] += 1
                
                vector_paths.append(path)

    # Save to JSON file
    output_file = 'vector_paths.json'
    with open(output_file, 'w') as f:
        json.dump(vector_paths, f, indent=4)
    print(f"Vector paths saved to {output_file}")

    print("\nCommand counts:")
    for cmd, count in sorted(command_counts.items()):
        print(f"  {cmd}: {count}")

if __name__ == "__main__":
    main()