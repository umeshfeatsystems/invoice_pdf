"""
Document Classifier Service
===========================
Uses Gemini AI to classify document types within a PDF.
Updated to support specific vendor routing (ABB/Crown).
"""

import google.generativeai as genai
from models.schemas import DocumentClassification
from models.gemini_config import get_generation_config
from utils.pdf_utils import get_pdf_page_count
from services.prompt_config import CLASSIFICATION_PROMPT


async def classify_documents(pdf_path: str, model_name: str = "gemini-2.5-flash") -> DocumentClassification:
    """
    Classify document types in a PDF and return page ranges.
    """
    # 1. Get PDF Info for Validation
    try:
        with open(pdf_path, "rb") as f:
            pdf_content = f.read()
        total_pages = get_pdf_page_count(pdf_content)
    except Exception:
        total_pages = 1000  # Fallback

    # 2. Call LLM
    pdf_file = genai.upload_file(pdf_path, mime_type="application/pdf")
    model = genai.GenerativeModel(model_name)
    generation_config = get_generation_config(response_schema=DocumentClassification)
    
    try:
        response = await model.generate_content_async(
            [CLASSIFICATION_PROMPT, pdf_file],
            generation_config=generation_config
        )
        raw_class = DocumentClassification.model_validate_json(response.text)
        
        # 3. SANITIZE OUTPUT
        def clean_ranges(ranges_list):
            cleaned = []
            for r in ranges_list:
                valid_pages = [p for p in r if isinstance(p, int) and 1 <= p <= total_pages]
                if valid_pages:
                    cleaned.append([min(valid_pages), max(valid_pages)])
            return cleaned

        # Clean all categories
        raw_class.invoices = clean_ranges(raw_class.invoices)
        raw_class.abb_invoices = clean_ranges(raw_class.abb_invoices)
        raw_class.crown_invoices = clean_ranges(raw_class.crown_invoices)
        raw_class.airway_bills = clean_ranges(raw_class.airway_bills)
        
        return raw_class

    except Exception as e:
        print(f"Classification Error: {e}")
        # Fallback: Assume whole doc is 1 global invoice
        return DocumentClassification(
            invoices=[[1, total_pages]], 
            abb_invoices=[],
            crown_invoices=[],
            airway_bills=[]
        )