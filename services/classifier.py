"""
Document Classifier Service
===========================
Uses Gemini AI to classify document types within a PDF.
"""

import google.generativeai as genai
from models.schemas import DocumentClassification
from models.gemini_config import get_generation_config
from utils.pdf_utils import get_pdf_page_count
from services.prompt_config import CLASSIFICATION_PROMPT


async def classify_documents(pdf_path: str, model_name: str = "gemini-2.5-pro") -> DocumentClassification:
    """
    Classify document types in a PDF and return page ranges.
    
    Args:
        pdf_path: Path to the PDF file
        model_name: Gemini model to use for classification
        
    Returns:
        DocumentClassification with page ranges for invoices and airway_bills
    """
    # 1. Get PDF Info for Validation
    try:
        with open(pdf_path, "rb") as f:
            pdf_content = f.read()
        total_pages = get_pdf_page_count(pdf_content)
    except Exception:
        total_pages = 1000  # Fallback if read fails

    # 2. Call LLM
    pdf_file = genai.upload_file(pdf_path, mime_type="application/pdf")
    model = genai.GenerativeModel(model_name)
    generation_config = get_generation_config(response_schema=DocumentClassification)
    
    try:
        response = model.generate_content(
            [CLASSIFICATION_PROMPT, pdf_file],
            generation_config=generation_config
        )
        raw_class = DocumentClassification.model_validate_json(response.text)
        
        # 3. SANITIZE OUTPUT (Critical Fix)
        # Remove garbage numbers (e.g. "7732") that are larger than total_pages
        
        def clean_ranges(ranges_list):
            cleaned = []
            for r in ranges_list:
                # Filter invalid pages
                valid_pages = [p for p in r if isinstance(p, int) and 1 <= p <= total_pages]
                if valid_pages:
                    # Convert to strict [start, end]
                    cleaned.append([min(valid_pages), max(valid_pages)])
            return cleaned

        raw_class.invoices = clean_ranges(raw_class.invoices)
        raw_class.airway_bills = clean_ranges(raw_class.airway_bills)
        
        return raw_class

    except Exception as e:
        print(f"Classification Error: {e}")
        # Fallback: Assume whole doc is 1 invoice if classification fails
        return DocumentClassification(invoices=[[1, total_pages]], airway_bills=[])