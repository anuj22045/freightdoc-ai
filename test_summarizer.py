
from modules.pdf_processor import extract_text_from_pdf

from modules.summarizer import summarize_document


PDF_PATH = "sample_documents/bill-of-lading.pdf"


# STEP 1: Extract PDF text
print("\nSTEP 1: Processing PDF...")

pdf_data = extract_text_from_pdf(PDF_PATH)

print("PDF:", pdf_data["filename"])

print("Total Pages:", pdf_data["total_pages"])


# STEP 2: Generate document summary
print("\nSTEP 2: Generating AI summary...")

summary = summarize_document(
    pdf_data["full_text"]
)


# STEP 3: Display summary
print("\nFREIGHTDOC AI - DOCUMENT SUMMARY")
print("=" * 60)

print(summary)

print("\n" + "=" * 60)

print("DOCUMENT SUMMARIZATION COMPLETED!")
