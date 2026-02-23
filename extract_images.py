import fitz # PyMuPDF
import os

# Specify the path to your PDF file
pdf_file_path = 'C:\\Users\\aalmuarik\\source\\repos\\ocr\\OCR-ensemble\\View Original Document.pdf'
doc = fitz.open(pdf_file_path)

# Create an output directory if it doesn't exist
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over each page in the document
for i in range(doc.page_count):
    page = doc.load_page(i) # Load a specific page

    # Render the page to a pixmap (image representation)
    # You can increase the dpi for higher resolution, e.g., get_pixmap(dpi=300)
    pix = page.get_pixmap()

    # Specify output path
    output = os.path.join(output_dir, f"pdfpage{i+1}.png")

    # Save the image
    pix.save(output)
    print(f"Saved {output}")

doc.close() # Close the document
print("Conversion successful!")
