import fitz  # PyMuPDF
import json
import argparse
from collections import defaultdict

def analyze_drawing(drawing, index):
    """Analyze a single drawing object."""
    items = drawing['items']
    cmd_types = [item[0] for item in items]
    return {
        'index': index,
        'stroke': drawing.get('stroke'),
        'fill': drawing.get('fill'),
        'width': drawing.get('width'),
        'commands': cmd_types,
        'num_items': len(items),
        'raw_items': items
    }

def main():
    parser = argparse.ArgumentParser(description="Debug vector graphics from PDF.")
    parser.add_argument('pdf_file', type=str, help="The PDF file to process")
    parser.add_argument('--page', type=int, default=0, help="Page number to extract")
    parser.add_argument('--sample', type=int, default=10, help="Number of drawings to analyze")
    args = parser.parse_args()

    doc = fitz.open(args.pdf_file)
    page = doc[args.page]
    drawings = page.get_drawings()

    print(f"Total drawings: {len(drawings)}")
    
    # Count raw commands
    raw_command_counts = defaultdict(int)
    for drawing in drawings:
        for item in drawing['items']:
            raw_command_counts[item[0]] += 1

    print("\nRaw command distribution:")
    for cmd, count in sorted(raw_command_counts.items()):
        print(f"  {cmd}: {count}")

    # Analyze sample drawings
    print(f"\nAnalyzing first {args.sample} drawings:")
    for i in range(min(args.sample, len(drawings))):
        analysis = analyze_drawing(drawings[i], i)
        print("\nDrawing", i)
        print("  Properties:", {k: v for k, v in analysis.items() if k not in ['raw_items', 'index']})
        print("  Raw items:")
        for item in analysis['raw_items']:
            print(f"    {item}")

if __name__ == "__main__":
    main()