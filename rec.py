import fitz
import json
from collections import defaultdict

def main():
    doc = fitz.open("drawings_page_8.pdf")  # Change to your filename
    page = doc[0]
    
    # Get text blocks
    text_data = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("lines"):
            for line in block["lines"]:
                if line.get("spans"):
                    for span in line["spans"]:
                        bbox = span["bbox"]
                        text_data.append({
                            "text": span["text"],
                            "bbox": bbox,
                            "font": span["font"],
                            "size": span["size"]
                        })

    # Get rectangles from drawings
    rect_data = []
    for drawing in page.get_drawings():
        items = drawing['items']
        if len(items) == 1 and items[0][0] == 're':
            rect = items[0][1]  # fitz.Rect object
            rect_data.append({
                "rect": [rect.x0, rect.y0, rect.width, rect.height],
                "stroke": drawing.get('stroke'),
                "fill": drawing.get('fill'),
                "width": drawing.get('width')
            })

    # Save diagnostic data
    diagnostic = {
        "text_blocks": text_data,
        "rectangles": rect_data
    }
    
    with open('diagnostic.json', 'w') as f:
        json.dump(diagnostic, f, indent=2)
        
    print(f"Found {len(text_data)} text blocks and {len(rect_data)} rectangles")
    
    # Print some sample data
    print("\nSample text blocks:")
    for block in text_data[:3]:
        print(f"Text: {block['text']}")
        print(f"BBox: {block['bbox']}")
        print(f"Font: {block['font']} Size: {block['size']}\n")
        
    print("\nSample rectangles:")
    for rect in rect_data[:3]:
        print(f"Rect: {rect['rect']}")
        print(f"Stroke: {rect['stroke']}")
        print(f"Fill: {rect['fill']}")
        print(f"Width: {rect['width']}\n")

if __name__ == "__main__":
    main()