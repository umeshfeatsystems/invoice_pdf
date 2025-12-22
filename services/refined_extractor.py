import logging
import asyncio
import google.generativeai as genai
from services.rate_limiter import get_rate_limiter, release_rate_limit

logger = logging.getLogger("logistics_refined")

async def verify_field_with_llm(
    pdf_path: str, 
    field_name: str, 
    field_description: str,
    valid_values: list = None,
    model_name: str = "gemini-2.5-flash"
) -> str:
    """
    Uses Gemini strictly as a refined extraction engine to find a specific field 
    directly from the PDF, replacing local OCR.
    """
    prompt = f"""
    You are a strict data extractor.
    
    Target Field: {field_name}
    Definition: {field_description}
    
    INSTRUCTIONS:
    1. Scan the provided document image/pdf visual strictly.
    2. LOCATE the specific value for the Target Field.
    3. Return ONLY the value extracted from the page.
    4. If the value is not present, is blank, or cannot be found, return 'null'.
    5. Do not add any markdown, JSON formatting, or explanation.
    """
    
    if valid_values:
        prompt += f"\n6. VALID VALUES ONLY: {', '.join(valid_values)}. If found value is not in this list, return 'null'."
    
    prompt += "\n\nExample Response: 12345\nExample Response: null"
    
    try:
        # Rate limit
        await get_rate_limiter().acquire(model_name)
        
        # Upload file directly for this specific check
        # (In a highly optimized flow, we might pass the already uploaded file handle, 
        # but for safety/isolation, we upload here for the distinct check).
        pdf_file = genai.upload_file(pdf_path, mime_type="application/pdf")
        
        model = genai.GenerativeModel(model_name)
        
        # Generate
        response = await model.generate_content_async(
            [prompt, pdf_file],
            request_options={"timeout": 60} # Fast timeout for single field
        )
        
        val = response.text.strip().replace("'", "").replace('"', "")
        if val.lower() == "null" or val == "":
            return None
            
        # Strict Code-Level Validation
        if valid_values:
            # Check for direct match or case-insensitive match
            if val in valid_values:
                return val
            
            # Simple cleanup check (e.g. if model returns "FCA Incoterms", try to find "FCA")
            for v in valid_values:
                if val == v or val.upper() == v:
                    return v
            
            logger.warning(f"Refined Extraction: extracted '{val}' not in valid list for {field_name}. Returning None.")
            return None
            
        return val
        
    except Exception as e:
        logger.error(f"Refined extraction (LLM) failed for {field_name}: {e}")
        return None
    finally:
        release_rate_limit(model_name)
