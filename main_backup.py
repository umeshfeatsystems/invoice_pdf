import os
import io
import time
import uuid
import logging
import asyncio
import shutil
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

import uvicorn
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from pypdf import PdfReader, PdfWriter
from dotenv import load_dotenv

# Import prompts from centralized config
from services.prompt_config import INVOICE_PROMPT, AWB_PROMPT, CLASSIFICATION_PROMPT

# ==========================================
# 1. CONFIGURATION & ENVIRONMENT
# ==========================================
load_dotenv()

def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("logistics_monolith")

logger = setup_logging()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables.")
else:
    genai.configure(api_key=API_KEY)

CONFIG = {
    "CLASSIFIER_MODEL": "gemini-2.5-flash",
    "EXTRACTOR_MODEL": "gemini-3-pro-preview"
}

PRICING = {
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-3-pro-preview": {"input": 3.50, "output": 10.50},
    "default": {"input": 3.50, "output": 10.50}
}

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================
def calculate_cost(model_name, input_tokens, output_tokens):
    pricing = PRICING.get("default")
    for key in PRICING:
        if key in model_name:
            pricing = PRICING[key]
            break
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)

def get_pdf_page_count(file_content: bytes) -> int:
    try:
        reader = PdfReader(io.BytesIO(file_content))
        return len(reader.pages)
    except Exception:
        return 0

def split_pdf(original_pdf_path: str, ranges: List[List[int]], output_dir: str, prefix: str) -> List[str]:
    reader = PdfReader(original_pdf_path)
    output_paths = []
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    for i, (start, end) in enumerate(ranges):
        writer = PdfWriter()
        if start < 1: start = 1
        if end > len(reader.pages): end = len(reader.pages)
        for page_num in range(start - 1, end):
            writer.add_page(reader.pages[page_num])
        output_filename = f"{prefix}_{i+1}_{start}-{end}.pdf"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "wb") as f:
            writer.write(f)
        output_paths.append(output_path)
    return output_paths

# ==========================================
# 3. PYDANTIC SCHEMAS (Updated field names)
# ==========================================
class InvoiceItem(BaseModel):
    """Line item from an invoice - field names match production data format."""
    item_no: Optional[str] = None
    item_description: Optional[str] = None
    item_part_no: Optional[str] = None
    item_po_no: Optional[str] = None
    
    # Tariff/Customs codes
    hsn_code: Optional[str] = Field(None, description="Harmonized System Code")
    item_cth: Optional[str] = Field(None, description="Customs Tariff Heading")
    item_ritc: Optional[str] = Field(None, description="Regional Import Tariff Code")
    item_ceth: Optional[str] = Field(None, description="Central Excise Tariff Heading")
    
    # Quantity and pricing (standardized naming)
    item_quantity: Optional[float] = None
    item_uom: Optional[str] = Field(None, description="Unit of Measure (PCS, KG, etc.)")
    item_unit_price: Optional[float] = None
    item_amount: Optional[float] = Field(None, description="Line total amount")
    
    # Origin details
    item_origin_country: Optional[str] = Field(None, description="Country of Origin")
    
    # Manufacturer details
    item_mfg_name: Optional[str] = Field(None, description="Manufacturer Name")
    item_mfg_addr: Optional[str] = Field(None, description="Manufacturer Address")
    item_mfg_country: Optional[str] = Field(None, description="Manufacturer Country")


class Invoice(BaseModel):
    """Invoice document structure - field names match production data format."""
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    
    # Parties
    seller_name: Optional[str] = Field(None, description="Supplier Name")
    seller_address: Optional[str] = None
    buyer_name: Optional[str] = None
    buyer_address: Optional[str] = None
    
    # Financial (invoice_currency for consistency)
    invoice_currency: Optional[str] = Field(None, description="Currency code (EUR, USD, etc.)")
    total_amount: Optional[float] = None
    invoice_exchange_rate: Optional[float] = None
    
    # Terms
    invoice_toi: Optional[str] = Field(None, description="Terms of Invoice (IncoTerms)")
    invoice_po_date: Optional[str] = None
    
    # Line items
    items: List[InvoiceItem] = Field(default_factory=list)


class AirwayBill(BaseModel):
    """Airway Bill document structure - field names match production data format."""
    master_awb_no: Optional[str] = None
    house_awb_no: Optional[str] = None
    shipper_name: Optional[str] = None
    pkg_in_qty: Optional[int] = Field(None, description="Number of packages")
    gross_weight_kg: Optional[float] = Field(None, description="Gross weight in KG")
    chargeable_weight_kg: Optional[float] = Field(None, description="Chargeable weight in KG")
    total_frieght: Optional[float] = Field(None, description="Total freight amount")
    frieght_currency: Optional[str] = Field(None, description="Currency for freight (EUR, SGD, etc.)")


class DocumentClassification(BaseModel):
    invoices: List[List[int]] = Field(default_factory=list)
    airway_bills: List[List[int]] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    document_type: str 
    page_range: List[int]
    data: Optional[dict] = None
    error: Optional[str] = None
    token_usage: Optional[dict] = None
    cost_usd: Optional[float] = 0.0


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    processing_time_seconds: Optional[float] = None
    file_name: Optional[str] = None
    model_used: Optional[str] = None
    classification: Optional[DocumentClassification] = None
    results: Optional[List[ExtractionResult]] = None
    total_tokens: Optional[int] = 0
    total_cost_usd: Optional[float] = 0.0
    error: Optional[str] = None

# ==========================================
# 4. AI GENERATION CONFIG HELPERS
# ==========================================
def resolve_refs(schema, defs=None):
    if defs is None: defs = schema.get("$defs", {}) or schema.get("definitions", {})
    if isinstance(schema, dict):
        if "$ref" in schema:
            ref = schema["$ref"]
            name = ref.split("/")[-1]
            if name in defs: return resolve_refs(defs[name], defs)
        new_schema = {}
        for k, v in schema.items():
            if k in ["$defs", "definitions"]: continue
            new_schema[k] = resolve_refs(v, defs)
        return new_schema
    elif isinstance(schema, list):
        return [resolve_refs(item, defs) for item in schema]
    return schema

def clean_schema(schema):
    schema = resolve_refs(schema)
    def _clean(s):
        if isinstance(s, dict):
            if "anyOf" in s:
                for opt in s["anyOf"]:
                    if opt.get("type") != "null": return _clean(opt)
            for key in ["default", "title", "$defs", "definitions"]:
                if key in s: del s[key]
            for key, value in s.items(): s[key] = _clean(value)
        elif isinstance(s, list):
            for i, item in enumerate(s): s[i] = _clean(item)
        return s
    return _clean(schema)

def get_generation_config(response_schema=None):
    config = {"response_mime_type": "application/json", "temperature": 0.0}
    if response_schema:
        try:
            if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
                raw_schema = response_schema.model_json_schema()
                config["response_schema"] = clean_schema(raw_schema)
            else:
                config["response_schema"] = response_schema
        except Exception as e:
            logger.error(f"Schema cleaning failed: {e}")
            config["response_schema"] = response_schema
    return config

# ==========================================
# 5. BUSINESS LOGIC & POST-PROCESSING
# ==========================================
async def execute_with_retry_async(func, retries=3, delay=2):
    last_err = None
    for i in range(retries):
        try:
            return await func()
        except Exception as e:
            logger.warning(f"Retry {i+1}: {e}")
            last_err = e
            await asyncio.sleep(delay)
    raise last_err

class PostProcessor:
    def process_invoice(self, data: Dict[str, Any]) -> Dict[str, Any]:
        toi = data.get("invoice_toi", "") or ""
        if toi and "FCA" in str(toi).upper(): data["invoice_toi"] = "FOB"
        if "items" in data and isinstance(data["items"], list):
            for item in data["items"]:
                desc = item.get("item_description", "") or ""
                part = item.get("item_part_no", "") or ""
                if part and part not in desc:
                    item["item_description"] = f"{desc} {part}".strip()
        return data

    def process_airway_bill(self, data: Dict[str, Any]) -> Dict[str, Any]:
        def clean_awb(value):
            if not value: return None
            return "".join(filter(str.isalnum, str(value)))
        if "master_awb_no" in data: data["master_awb_no"] = clean_awb(data["master_awb_no"])
        if "house_awb_no" in data: data["house_awb_no"] = clean_awb(data["house_awb_no"])
        return data

post_processor = PostProcessor()

# ==========================================
# 6. AI SERVICE FUNCTIONS
# ==========================================
async def classify_documents(pdf_path: str, model_name: str) -> DocumentClassification:
    try:
        with open(pdf_path, "rb") as f:
            total_pages = get_pdf_page_count(f.read())
    except: total_pages = 1000

    pdf_file = genai.upload_file(pdf_path, mime_type="application/pdf")
    model = genai.GenerativeModel(model_name)
    config = get_generation_config(response_schema=DocumentClassification)
    
    try:
        response = await model.generate_content_async(
            [CLASSIFICATION_PROMPT, pdf_file],
            generation_config=config,
            request_options={"timeout": 600}
        )
        raw_class = DocumentClassification.model_validate_json(response.text)
        
        def clean_ranges(ranges_list):
            cleaned = []
            for r in ranges_list:
                valid_pages = [p for p in r if isinstance(p, int) and 1 <= p <= total_pages]
                if valid_pages: cleaned.append([min(valid_pages), max(valid_pages)])
            return cleaned

        raw_class.invoices = clean_ranges(raw_class.invoices)
        raw_class.airway_bills = clean_ranges(raw_class.airway_bills)
        return raw_class
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return DocumentClassification(invoices=[[1, total_pages]], airway_bills=[])

async def extract_data(pdf_path: str, model_name: str, doc_type: str) -> Tuple[Any, Optional[dict]]:
    pdf_file = genai.upload_file(pdf_path, mime_type="application/pdf")
    model = genai.GenerativeModel(model_name)
    
    if doc_type == "invoice":
        prompt = INVOICE_PROMPT
        schema = Invoice
    else:
        prompt = AWB_PROMPT
        schema = AirwayBill
        
    config = get_generation_config(response_schema=schema)
    async def _call():
        return await model.generate_content_async(
            [prompt, pdf_file], 
            generation_config=config, 
            request_options={"timeout": 600}
        )
    
    try:
        response = await execute_with_retry_async(_call)
        usage = None
        if response.usage_metadata:
            usage = {
                "prompt_token_count": response.usage_metadata.prompt_token_count,
                "candidates_token_count": response.usage_metadata.candidates_token_count,
                "total_token_count": response.usage_metadata.total_token_count
            }
        return schema.model_validate_json(response.text), usage
    except Exception as e:
        logger.error(f"Extraction failed for {doc_type}: {e}")
        return schema(), None

# ==========================================
# 7. FASTAPI APPLICATION & LOGIC
# ==========================================
app = FastAPI(title="Logistics Monolith API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS = {}
UPLOAD_DIR = "uploads"
SPLIT_DIR = "splits"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation Error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "message": "Check 'file' field name."},
    )

async def run_pipeline(job_id: str, file_path: str, model_name: str):
    """Executes the full pipeline and updates the JOBS dict."""
    try:
        JOBS[job_id]["status"] = "classifying"
        start_time = time.time()
        
        logger.info(f"Job {job_id}: Classifying...")
        classification = await classify_documents(file_path, CONFIG["CLASSIFIER_MODEL"])
        JOBS[job_id]["classification"] = classification.dict()
        
        JOBS[job_id]["status"] = "splitting"
        split_tasks = []
        for r in classification.invoices: split_tasks.append({"type": "invoice", "range": r})
        for r in classification.airway_bills: split_tasks.append({"type": "airway_bill", "range": r})
        
        ranges = [t["range"] for t in split_tasks]
        if not ranges:
             split_tasks = [{"type": "invoice", "range": [1, 1]}]
             split_paths = [file_path] 
        else:
             split_paths = split_pdf(file_path, ranges, os.path.join(SPLIT_DIR, job_id), job_id)
        
        JOBS[job_id]["status"] = "extracting"
        results = []
        total_tokens = 0
        total_cost = 0.0
        
        for i, split_path in enumerate(split_paths):
            if i >= len(split_tasks): break
            doc_type = split_tasks[i]["type"]
            page_range = split_tasks[i]["range"]
            
            pydantic_obj, usage = await extract_data(split_path, model_name, doc_type)
            extracted_data = pydantic_obj.dict()
            
            if doc_type == "invoice":
                extracted_data = post_processor.process_invoice(extracted_data)
            elif doc_type == "airway_bill":
                extracted_data = post_processor.process_airway_bill(extracted_data)
                
            if usage:
                cost = calculate_cost(model_name, usage.get("prompt_token_count", 0), usage.get("candidates_token_count", 0))
                total_tokens += usage.get("total_token_count", 0)
                total_cost += cost
            
            results.append(ExtractionResult(
                document_type=doc_type,
                page_range=page_range,
                data=extracted_data,
                token_usage=usage,
                cost_usd=total_cost
            ))
            
        JOBS[job_id]["results"] = [r.dict() for r in results]
        JOBS[job_id]["total_tokens"] = total_tokens
        JOBS[job_id]["total_cost_usd"] = total_cost
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["completed_at"] = datetime.now()
        JOBS[job_id]["processing_time_seconds"] = time.time() - start_time
        
    except Exception as e:
        logger.error(f"Job {job_id} Failed: {e}")
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)

@app.post("/api/v1/process-document", response_model=JobStatusResponse)
async def process_document(file: UploadFile = File(...)):
    """
    Synchronous Endpoint: Uploads file -> Waits for Processing -> Returns Final JSON
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF allowed.")

    job_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}.pdf")
    
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Upload failed.")
        
    # Initialize Job Entry
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now(),
        "file_name": file.filename,
        "model_used": CONFIG["EXTRACTOR_MODEL"]
    }
    
    # ---------------------------------------------------------
    # BLOCKING CALL: We await the entire pipeline here.
    # The client will wait until this function finishes.
    # ---------------------------------------------------------
    await run_pipeline(job_id, file_path, CONFIG["EXTRACTOR_MODEL"])
    
    # Return the final result immediately
    return JobStatusResponse(**JOBS[job_id])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=300)