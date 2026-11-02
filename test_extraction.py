
from modules.pdf_processor import extract_text_from_pdf

from modules.information_extractor import (
    extract_shipment_information
)


PDF_PATH = "sample_documents/bill-of-lading.pdf"


# STEP 1: Extract PDF text
print("\nSTEP 1: Processing PDF...")

pdf_data = extract_text_from_pdf(PDF_PATH)

print("PDF:", pdf_data["filename"])


# STEP 2: Extract structured information
print("\nSTEP 2: Extracting shipment information...")

shipment_data = extract_shipment_information(
    pdf_data["full_text"]
)


# STEP 3: Display extracted information
print("\nSHIPMENT INFORMATION")
print("=" * 50)

for key, value in shipment_data.items():

    if key == "cargo_items":

        print("\nCARGO ITEMS:")

        for index, cargo in enumerate(value, start=1):

            print(f"\nCargo {index}:")

            for cargo_key, cargo_value in cargo.items():

                print(f"  {cargo_key}: {cargo_value}")

    else:

        print(f"{key}: {value}")


print("\nINFORMATION EXTRACTION COMPLETED!")
