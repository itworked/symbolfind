import fitz  # PyMuPDF
import os

def split_pdf_to_pages(pdf_path):
    # Open the PDF file
    with fitz.open(pdf_path) as pdf:
        # Get the base name of the PDF without extension
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]

        # Loop through each page
        for page_num in range(pdf.page_count):
            # Select the page
            page = pdf.load_page(page_num)
            
            # Create a new PDF document with this page
            new_pdf = fitz.open()
            new_pdf.insert_pdf(pdf, from_page=page_num, to_page=page_num)
            
            # Define the output filename
            output_filename = f"{base_name}_page_{page_num + 1}.pdf"
            
            # Save the page as a new PDF
            new_pdf.save(output_filename)
            new_pdf.close()

            print(f"Saved: {output_filename}")

# Usage
pdf_path = "drawings.pdf"  # Replace with your PDF file path
split_pdf_to_pages(pdf_path)
