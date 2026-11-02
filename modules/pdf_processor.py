from pypdf import PdfReader
from pathlib import Path

class PDFProcessingError(Exception):
    """Custom Exception for PDF processing errors."""
    pass

def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        if reader.is_encrypted:
            raise PDFProcessingError(
                "this pdf is password protected",
                "upload and unprotected PDF"
            )

        pages_data=[]
        full_text=[]

        for page_number, page in enumerate(reader.pages, start=1):
            extracted_text = page.extract_text() or ""
            pages_data.append({
                "page_number":page_number,
                "text":extracted_text.strip()
            })

            if extracted_text.strip():
                full_text.append(extracted_text.strip())

        combined_text = "\n\n".join(full_text)

        if not combined_text.strip():
            raise PDFProcessingError(
                "No readable text found in this PDF. "
                "It may be a scanned document requiring OCR."
            )

        #get file name
        filename= getattr(
            pdf_file, "name",
            Path(pdf_file).name if isinstance(
                pdf_file, (str, Path)
            )else "uploaded_document.pdf"
        )

        return {
            "filename": filename,
            "total_pages": len(reader.pages),
            "pages": pages_data,
            "full_text": combined_text,
            "total_characters": len(combined_text)
        }

    except PDFProcessingError:
        raise
    except Exception as e:
        raise PDFProcessingError(
            f"Unable to process PDF: {str(e)}"
        ) from e