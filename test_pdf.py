
from modules.pdf_processor import (
    extract_text_from_pdf,
    PDFProcessingError
)

PDF_PATH = "sample_documents/bill-of-lading.pdf"


try:
    result = extract_text_from_pdf(PDF_PATH)

    print("\nPDF PROCESSING SUCCESSFUL")
    print("-------------------------")

    print("Filename:", result["filename"])
    print("Total Pages:", result["total_pages"])
    print("Total Characters:", result["total_characters"])

    print("\nEXTRACTED TEXT:")
    print("-------------------------")

    for page in result["pages"]:

        print(f"\n--- Page {page['page_number']} ---")
        print(page["text"][:1000])

except PDFProcessingError as e:
    print("\nPDF PROCESSING FAILED")
    print("Error:", e)
