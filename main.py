"""
Optimized Logistics Document Processing Pipeline
=================================================
Key Features:
1. Specific Vendor Routing (Commin, Type1, Type2, Crown, ABB, Global)
2. Parallel extraction (asyncio.gather) with Concurrency Control
3. Dynamic Prompt Selection based on Classification
4. Robust Error Handling for API Instability
"""

import os
import io
import time
import uuid
import logging
import asyncio
import aiofiles
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

import uvicorn
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader, PdfWriter
from dotenv import load_dotenv

# Import prompts from centralized config
from services.prompt_config import (
    INVOICE_PROMPT, 
    ABB_PROMPT, 
    CROWN_PROMPT,
    COMMIN_PROMPT,
    TYPE1_PROMPT,
    TYPE2_PROMPT, 
    AWB_PROMPT, 
    CLASSIFICATION_PROMPT,
    REFINED_EXTRACTION_CONFIG
)
from services.refined_extractor import verify_field_with_llm

# Import enhanced rate limiter with release function
from services.rate_limiter import initialize_rate_limiter, get_rate_limiter, release_rate_limit, get_rate_limit_stats

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
    return logging.getLogger("logistics_optimized")

logger = setup_logging()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables.")
else:
    genai.configure(api_key=API_KEY)

CONFIG = {
    # === MODEL CONFIGURATION ===
    # ONLY use these models - they have favorable rate limits
    # gemini-2.5-flash: 1000 RPM (classification) 
    # gemini-2.5-pro: 150 RPM (extraction)
    "CLASSIFIER_MODEL": "gemini-2.5-flash",    # Fast classification
    "EXTRACTOR_MODEL": "gemini-2.5-pro",       # Accurate extraction
    
    # === TIMEOUT CONFIGURATION ===
    "CLASSIFICATION_TIMEOUT": 120,           
    "BASE_TIMEOUT_SECONDS": 180,             
    "TIMEOUT_PER_PAGE": 30,                  
    "MAX_TIMEOUT_SECONDS": 1200,             
    
    # === RATE LIMIT SAFE PARALLEL CONFIGURATION ===
    # With gemini-2.5-pro at 150 RPM (effective 105 with 70% margin):
    # - Max 1.75 requests/second
    # - Minimum 571ms between requests
    # For 40+ splits: keeps us well under limits
    "MAX_CONCURRENT_EXTRACTIONS": 2,         # Only 2 concurrent to avoid bursts
    "BATCH_SIZE": 2,                         # Process 2 at a time
    "INTER_BATCH_DELAY_SECONDS": 3.0,        # Reduced from 5s (rate limiter handles pacing)
    "MIN_DELAY_BETWEEN_CALLS": 1.0,          # Backup delay (rate limiter is primary)
    
    # === ROBUST RETRY CONFIGURATION ===
    "MAX_RETRIES": 5,                        
    "INITIAL_RETRY_DELAY": 10.0,             
    "MAX_RETRY_DELAY": 60.0,                 
    "RETRY_MULTIPLIER": 2.0,                 
}

PRICING = {
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 3.75}, 
    "default": {"input": 0.10, "output": 0.40}
}

QUOTA_ERROR_PATTERNS = [
    "quota", "rate limit", "resource exhausted", "429",
    "too many requests", "exceeded", "limit exceeded",
    "503", "service unavailable", "internal server error",
    "504", "cancelled", "deadline exceeded"
]

thread_pool = ThreadPoolExecutor(max_workers=4)
_api_semaphore = None

def get_api_semaphore():
    global _api_semaphore
    if _api_semaphore is None:
        _api_semaphore = asyncio.Semaphore(CONFIG["MAX_CONCURRENT_EXTRACTIONS"])
    return _api_semaphore

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

def calculate_timeout_for_pages(num_pages: int) -> int:
    base = CONFIG["BASE_TIMEOUT_SECONDS"]
    per_page = CONFIG["TIMEOUT_PER_PAGE"]
    max_timeout = CONFIG["MAX_TIMEOUT_SECONDS"]
    return min(base + (num_pages * per_page), max_timeout)

def get_pdf_page_count(file_content: bytes) -> int:
    try:
        reader = PdfReader(io.BytesIO(file_content))
        return len(reader.pages)
    except Exception:
        return 0

def split_pdf_sync(original_pdf_path: str, ranges: List[List[int]], output_dir: str, prefix: str) -> List[str]:
    reader = PdfReader(original_pdf_path)
    output_paths = []
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    
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

async def split_pdf_async(original_pdf_path: str, ranges: List[List[int]], output_dir: str, prefix: str) -> List[str]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        thread_pool, 
        split_pdf_sync, 
        original_pdf_path, ranges, output_dir, prefix
    )

def is_transient_error(error: Exception) -> bool:
    error_str = str(error).lower()
    return any(pattern in error_str for pattern in QUOTA_ERROR_PATTERNS)

async def execute_with_exponential_backoff(
    async_func,
    max_retries: int = None,
    initial_delay: float = None,
    max_delay: float = None,
    multiplier: float = None,
    operation_name: str = "API call"
):
    max_retries = max_retries or CONFIG["MAX_RETRIES"]
    initial_delay = initial_delay or CONFIG["INITIAL_RETRY_DELAY"]
    max_delay = max_delay or CONFIG["MAX_RETRY_DELAY"]
    multiplier = multiplier or CONFIG["RETRY_MULTIPLIER"]
    
    delay = initial_delay
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                await asyncio.sleep(CONFIG["MIN_DELAY_BETWEEN_CALLS"])
            return await async_func()
        except Exception as e:
            if attempt >= max_retries:
                logger.error(f"[{operation_name}] Failed after {max_retries} retries: {e}")
                raise e
            if is_transient_error(e):
                logger.warning(
                    f"[{operation_name}] Transient Error (attempt {attempt+1}/{max_retries}). "
                    f"Waiting {delay:.1f}s... Error: {str(e)[:100]}"
                )
                jitter = delay * 0.2 * (random.random() * 2 - 1)
                await asyncio.sleep(delay + jitter)
                delay = min(delay * multiplier, max_delay)
            else:
                raise e

# ==========================================
# 3. PYDANTIC SCHEMAS
# ==========================================
class InvoiceItem(BaseModel):
    item_no: Optional[str] = None
    item_description: Optional[str] = None
    item_part_no: Optional[str] = None
    item_po_no: Optional[str] = None
    item_date: Optional[str] = Field(None, description="Date specific to the line item (YYYY-MM-DD)")
    hsn_code: Optional[str] = Field(None, description="Harmonized System Code")
    item_cth: Optional[str] = Field(None, description="Customs Tariff Heading")
    item_ritc: Optional[str] = Field(None, description="Regional Import Tariff Code")
    item_ceth: Optional[str] = Field(None, description="Central Excise Tariff Heading")
    item_quantity: Optional[float] = None
    item_uom: Optional[str] = Field(None, description="Unit of Measure (PCS, KG, etc.)")
    item_unit_price: Optional[float] = None
    item_amount: Optional[float] = Field(None, description="Line total amount")
    item_origin_country: Optional[str] = Field(None, description="Country of Origin")
    item_mfg_name: Optional[str] = Field(None, description="Manufacturer Name")
    item_mfg_addr: Optional[str] = Field(None, description="Manufacturer Address")
    item_mfg_country: Optional[str] = Field(None, description="Manufacturer Country")

class Invoice(BaseModel):
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    seller_name: Optional[str] = Field(None, description="Supplier Name")
    seller_address: Optional[str] = None
    buyer_name: Optional[str] = None
    buyer_address: Optional[str] = None
    invoice_currency: Optional[str] = Field(None, description="Currency code (EUR, USD, etc.)")
    total_amount: Optional[float] = None
    invoice_exchange_rate: Optional[float] = None
    invoice_toi: Optional[str] = Field(None, description="Terms of Invoice (IncoTerms)")
    invoice_po_date: Optional[str] = None
    items: List[InvoiceItem] = Field(default_factory=list)

class AirwayBill(BaseModel):
    master_awb_no: Optional[str] = None
    house_awb_no: Optional[str] = None
    shipper_name: Optional[str] = None
    pkg_in_qty: Optional[int] = Field(None, description="Number of packages")
    gross_weight_kg: Optional[float] = Field(None, description="Gross weight in KG")
    chargeable_weight_kg: Optional[float] = Field(None, description="Chargeable weight in KG")
    total_frieght: Optional[float] = Field(None, description="Total freight amount")
    frieght_currency: Optional[str] = Field(None, description="Currency for freight (EUR, SGD, etc.)")

class DocumentClassification(BaseModel):
    """
    Classification result showing page ranges for each document type.
    Now supports 5 specific vendors + Global fallback.
    """
    # 1. Commin (SIN-10076768)
    commin_invoices: List[List[int]] = Field(default_factory=list, description="Commin Invoices (SIN-...)")
    
    # 2. Type 1 (0100935473)
    type1_invoices: List[List[int]] = Field(default_factory=list, description="Type 1 Invoices (0100935473...)")
    
    # 3. Type 2 (KM_558...)
    type2_invoices: List[List[int]] = Field(default_factory=list, description="Type 2 Invoices (KM_...)")
    
    # 4. Crown
    crown_invoices: List[List[int]] = Field(default_factory=list, description="Crown Worldwide Invoices")
    
    # 5. ABB
    abb_invoices: List[List[int]] = Field(default_factory=list, description="ABB or Epiroc Invoices")
    
    # Fallback
    invoices: List[List[int]] = Field(default_factory=list, description="Standard/Global Invoices")
    
    # Logistics
    airway_bills: List[List[int]] = Field(default_factory=list)

class ExtractionResult(BaseModel):
    document_type: str 
    page_range: List[int]
    data: Optional[dict] = None
    error: Optional[str] = None
    token_usage: Optional[dict] = None
    cost_usd: Optional[float] = 0.0
    extraction_time_seconds: Optional[float] = None

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
    classification_time_seconds: Optional[float] = None
    extraction_parallelism: Optional[int] = None

# ==========================================
# 4. AI CONFIG HELPERS
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
        except Exception:
            config["response_schema"] = response_schema
    return config

# ==========================================
# 5. SERVICE FUNCTIONS
# ==========================================

# Pre-create model instances (reusable)
_classifier_model = None
_extractor_model = None

def get_classifier_model():
    global _classifier_model
    if _classifier_model is None:
        _classifier_model = genai.GenerativeModel(CONFIG["CLASSIFIER_MODEL"])
    return _classifier_model

def get_extractor_model():
    global _extractor_model
    if _extractor_model is None:
        _extractor_model = genai.GenerativeModel(CONFIG["EXTRACTOR_MODEL"])
    return _extractor_model

async def classify_documents_optimized(pdf_path: str) -> Tuple[DocumentClassification, float]:
    """Optimized classification with timing."""
    start_time = time.time()
    
    try:
        with open(pdf_path, "rb") as f:
            total_pages = get_pdf_page_count(f.read())
    except: 
        total_pages = 1000

    pdf_file = genai.upload_file(pdf_path, mime_type="application/pdf")
    model = get_classifier_model()
    
    config = get_generation_config(response_schema=DocumentClassification)
    
    try:
        # Rate limit before classification
        await get_rate_limiter().acquire(CONFIG["CLASSIFIER_MODEL"])

        response = await model.generate_content_async(
            [CLASSIFICATION_PROMPT, pdf_file],
            generation_config=config,
            request_options={"timeout": CONFIG["CLASSIFICATION_TIMEOUT"]}
        )
        raw_class = DocumentClassification.model_validate_json(response.text)
        
        # Sanitize output
        def clean_ranges(ranges_list):
            cleaned = []
            for r in ranges_list:
                valid_pages = [p for p in r if isinstance(p, int) and 1 <= p <= total_pages]
                if valid_pages: cleaned.append([min(valid_pages), max(valid_pages)])
            return cleaned

        raw_class.commin_invoices = clean_ranges(raw_class.commin_invoices)
        raw_class.type1_invoices = clean_ranges(raw_class.type1_invoices)
        raw_class.type2_invoices = clean_ranges(raw_class.type2_invoices)
        raw_class.crown_invoices = clean_ranges(raw_class.crown_invoices)
        raw_class.abb_invoices = clean_ranges(raw_class.abb_invoices)
        raw_class.invoices = clean_ranges(raw_class.invoices)
        raw_class.airway_bills = clean_ranges(raw_class.airway_bills)
        
        elapsed = time.time() - start_time
        return raw_class, elapsed
        
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        elapsed = time.time() - start_time
        return DocumentClassification(invoices=[[1, total_pages]], airway_bills=[]), elapsed


async def extract_single_document(
    pdf_path: str, 
    doc_type: str, 
    page_range: List[int],
    doc_index: int
) -> ExtractionResult:
    """
    Extract a single document using the correct prompt based on doc_type.
    """
    start_time = time.time()
    num_pages = page_range[1] - page_range[0] + 1 if len(page_range) == 2 else 1
    dynamic_timeout = calculate_timeout_for_pages(num_pages)
    
    # 1. SELECT PROMPT & SCHEMA
    if doc_type == "commin_invoice":
        prompt = COMMIN_PROMPT
        schema = Invoice
        log_type = "Commin Invoice"
    elif doc_type == "type1_invoice":
        prompt = TYPE1_PROMPT
        schema = Invoice
        log_type = "Type 1 Invoice"
    elif doc_type == "type2_invoice":
        prompt = TYPE2_PROMPT
        schema = Invoice
        log_type = "Type 2 Invoice"
    elif doc_type == "abb_invoice":
        prompt = ABB_PROMPT
        schema = Invoice
        log_type = "ABB/Epiroc Invoice"
    elif doc_type == "crown_invoice":
        prompt = CROWN_PROMPT
        schema = Invoice
        log_type = "Crown Invoice"
    elif doc_type == "invoice":
        prompt = INVOICE_PROMPT
        schema = Invoice
        log_type = "Global Invoice"
    else:
        prompt = AWB_PROMPT
        schema = AirwayBill
        log_type = "Airway Bill"
    
    logger.info(f"Doc {doc_index}: Starting {log_type} (pages {page_range[0]}-{page_range[1]}, {num_pages} pages, timeout={dynamic_timeout}s)")
    
    async with get_api_semaphore():  # Concurrency control via semaphore
        try:
            async def do_extraction():
                # Rate limit before extraction (sliding window + burst protection)
                await get_rate_limiter().acquire(CONFIG["EXTRACTOR_MODEL"])

                pdf_file = genai.upload_file(pdf_path, mime_type="application/pdf")
                model = get_extractor_model()
                
                config = get_generation_config(response_schema=schema)
                
                response = await model.generate_content_async(
                    [prompt, pdf_file],
                    generation_config=config,
                    request_options={"timeout": dynamic_timeout}
                )
                return response, schema
            
            # Execute with exponential backoff retry
            response, schema = await execute_with_exponential_backoff(
                do_extraction,
                operation_name=f"Extract doc {doc_index} ({doc_type})"
            )
            
            usage = None
            if response.usage_metadata:
                usage = {
                    "prompt_token_count": response.usage_metadata.prompt_token_count,
                    "candidates_token_count": response.usage_metadata.candidates_token_count,
                    "total_token_count": response.usage_metadata.total_token_count
                }
            
            pydantic_obj = schema.model_validate_json(response.text)
            extracted_data = pydantic_obj.dict()
            
            # --- START REFINED EXTRACTION LOGIC ---
            # Check if this document type has fields requiring secondary text-based verification
            if doc_type in REFINED_EXTRACTION_CONFIG:
                logger.info(f"Doc {doc_index}: Running refined extraction for {doc_type}...")
                
                for field_cfg in REFINED_EXTRACTION_CONFIG[doc_type]:
                    if field_cfg.get("enabled", False):
                        field_name = field_cfg["field"]
                            
                        # Secondary Verification Call (LLM-based per field)
                        verified_value = await verify_field_with_llm(
                            pdf_path,  # Pass the file path directly
                            field_name,
                            field_cfg["description"],
                            valid_values=field_cfg.get("valid_values"), # Pass strict validation list
                            model_name=CONFIG["CLASSIFIER_MODEL"] # Use Flash for speed/cost
                        )
                        
                        logger.info(f"Doc {doc_index}: Refined {field_name} -> {verified_value}")
                        
                        # Update data (override or set)
                        extracted_data[field_name] = verified_value
            # --- END REFINED EXTRACTION LOGIC ---

            if schema == Invoice:
                extracted_data = post_process_invoice(extracted_data)
            else:
                extracted_data = post_process_airway_bill(extracted_data)
            
            elapsed = time.time() - start_time
            cost = calculate_cost(
                CONFIG["EXTRACTOR_MODEL"], 
                usage.get("prompt_token_count", 0) if usage else 0, 
                usage.get("candidates_token_count", 0) if usage else 0
            )
            
            logger.info(f"Doc {doc_index} ({doc_type}) extracted in {elapsed:.2f}s")
            
            return ExtractionResult(
                document_type=doc_type,
                page_range=page_range,
                data=extracted_data,
                token_usage=usage,
                cost_usd=cost,
                extraction_time_seconds=round(elapsed, 2)
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Extraction failed for doc {doc_index} after all retries: {e}")
            return ExtractionResult(
                document_type=doc_type,
                page_range=page_range,
                error=str(e),
                extraction_time_seconds=round(elapsed, 2)
            )
        finally:
            # CRITICAL: Release rate limiter slot when extraction completes
            release_rate_limit(CONFIG["EXTRACTOR_MODEL"])


def post_process_invoice(data: Dict[str, Any]) -> Dict[str, Any]:
    # Valid Incoterms 2020 - only these values are accepted
    VALID_INCOTERMS = {"EXW", "FCA", "CPT", "CIP", "DAP", "DPU", "DDP", "FAS", "FOB", "CFR", "CIF"}
    INVALID_LITERALS = {"", "null", "string", "number", "integer", "float", "boolean", "None"}
    
    def clean_value(value):
        if value is None: return None
        if isinstance(value, str) and value.strip() in INVALID_LITERALS: return None
        return value
    
    # 1. Exchange Rate Cleanup (1.0 -> None)
    if str(data.get("invoice_exchange_rate", "")).replace(".0", "") == "1":
        data["invoice_exchange_rate"] = None
    
    # TOI Logic REMOVED per user request ("extraction only")
    # if "invoice_toi" in data:
    #     data["invoice_toi"] = validate_incoterm(data.get("invoice_toi"))
    
    if "items" in data and isinstance(data["items"], list):
        for idx, item in enumerate(data["items"], 1):
            # Normalize UOM to uppercase
            if item.get("item_uom"):
                item["item_uom"] = str(item["item_uom"]).upper().strip()
            
            # 2. Extract Fields
            desc = (item.get("item_description") or "").strip()
            part = (item.get("item_part_no") or "").strip()
            origin = (item.get("item_origin_country") or "").strip()

            # 3. Description Cleaning (Remove Part Number if present at end)
            if desc and part and desc.endswith(part):
                clean_desc = desc[:-len(part)].strip()
                if clean_desc:
                    item["item_description"] = clean_desc
            
            # 4. Standardize Crown Origin
            if origin in ["United States", "Germany", "China", "Mexico"]:
                item["item_origin_country"] = f"Made in {origin}"
            
            # Clean all values
            for key in list(item.keys()): 
                item[key] = clean_value(item[key])
    
    for key in list(data.keys()):
        if key != "items": data[key] = clean_value(data[key])
    return data

def post_process_airway_bill(data: Dict[str, Any]) -> Dict[str, Any]:
    def clean_mawb(value):
        if not value: return None
        digits = "".join(filter(str.isdigit, str(value)))
        return digits if digits else None
    
    if "master_awb_no" in data: data["master_awb_no"] = clean_mawb(data.get("master_awb_no"))
    if "house_awb_no" in data and data["house_awb_no"]:
        data["house_awb_no"] = "".join(filter(str.isalnum, str(data["house_awb_no"])))
    
    for field in ["pkg_in_qty", "gross_weight_kg", "chargeable_weight_kg", "total_frieght"]:
        if field in data and isinstance(data[field], str):
            try: data[field] = float(data[field].replace(",", "."))
            except: pass
            
    for key, value in data.items():
        if value == "" or value == "null": data[key] = None
    return data

# ==========================================
# 6. OPTIMIZED PIPELINE (PARALLEL + DASHBOARD)
# ==========================================
async def run_pipeline_optimized(job_id: str, file_path: str, model_name: str):
    """
    Optimized pipeline with PARALLEL extraction and Terminal Dashboard.
    """
    try:
        JOBS[job_id]["status"] = "classifying"
        pipeline_start = time.time()
        
        # Step 1: Classification
        logger.info(f"Job {job_id}: Classifying...")
        classification, class_time = await classify_documents_optimized(file_path)
        JOBS[job_id]["classification"] = classification.dict()
        JOBS[job_id]["classification_time_seconds"] = round(class_time, 2)
        
        # ===== DASHBOARD: CLASSIFICATION =====
        def count_pages(ranges): return sum((r[1] - r[0] + 1) for r in ranges) if ranges else 0
        
        print("\n" + "="*60)
        print(f"📋 CLASSIFICATION RESULT (Job: {job_id[:8]}...)")
        print("="*60)
        print(f"⏱️  Classification time: {class_time:.2f}s")
        print(f"\n📦 INVOICES (Standard): {len(classification.invoices)} docs ({count_pages(classification.invoices)} pages)")
        print(f"🛠️  COMMIN INVOICES:     {len(classification.commin_invoices)} docs ({count_pages(classification.commin_invoices)} pages)")
        print(f"📄 TYPE 1 INVOICES:     {len(classification.type1_invoices)} docs ({count_pages(classification.type1_invoices)} pages)")
        print(f"📄 TYPE 2 INVOICES:     {len(classification.type2_invoices)} docs ({count_pages(classification.type2_invoices)} pages)")
        print(f"🔧 ABB/EPIROC:          {len(classification.abb_invoices)} docs ({count_pages(classification.abb_invoices)} pages)")
        print(f"👑 CROWN INVOICES:      {len(classification.crown_invoices)} docs ({count_pages(classification.crown_invoices)} pages)")
        print(f"✈️  AIRWAY BILLS:        {len(classification.airway_bills)} docs ({count_pages(classification.airway_bills)} pages)")
        print("="*60 + "\n")
        
        # Step 2: Split PDF & Build Tasks
        JOBS[job_id]["status"] = "splitting"
        split_tasks = []
        
        # Priority: Specific -> Global -> AWB
        for r in classification.commin_invoices: split_tasks.append({"type": "commin_invoice", "range": r})
        for r in classification.type1_invoices:  split_tasks.append({"type": "type1_invoice", "range": r})
        for r in classification.type2_invoices:  split_tasks.append({"type": "type2_invoice", "range": r})
        for r in classification.abb_invoices:    split_tasks.append({"type": "abb_invoice", "range": r})
        for r in classification.crown_invoices:  split_tasks.append({"type": "crown_invoice", "range": r})
        for r in classification.invoices:        split_tasks.append({"type": "invoice", "range": r})
        for r in classification.airway_bills:    split_tasks.append({"type": "airway_bill", "range": r})
        
        ranges = [t["range"] for t in split_tasks]
        if not ranges:
            split_tasks = [{"type": "invoice", "range": [1, 1]}]
            split_paths = [file_path]
        else:
            split_paths = await split_pdf_async(
                file_path, ranges, 
                os.path.join(SPLIT_DIR, job_id), job_id
            )
        
        # Step 3: PARALLEL EXTRACTION
        JOBS[job_id]["status"] = "extracting"
        JOBS[job_id]["extraction_parallelism"] = min(len(split_paths), CONFIG["MAX_CONCURRENT_EXTRACTIONS"])
        
        extraction_start = time.time()
        batch_size = CONFIG["BATCH_SIZE"]
        inter_batch_delay = CONFIG["INTER_BATCH_DELAY_SECONDS"]
        
        extraction_tasks = [
            {
                "pdf_path": split_paths[i],
                "doc_type": split_tasks[i]["type"],
                "page_range": split_tasks[i]["range"],
                "doc_index": i + 1
            }
            for i in range(len(split_paths))
            if i < len(split_tasks)
        ]
        
        all_results = []
        num_batches = (len(extraction_tasks) + batch_size - 1) // batch_size
        
        logger.info(f"Job {job_id}: Processing {len(extraction_tasks)} documents in {num_batches} batches (Parallelism={CONFIG['MAX_CONCURRENT_EXTRACTIONS']})")
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(extraction_tasks))
            batch = extraction_tasks[batch_start:batch_end]
            
            logger.info(f"Job {job_id}: Starting batch {batch_idx + 1}/{num_batches} ({len(batch)} docs)")
            
            # [FIXED] PARALLEL EXECUTION
            # 1. Create list of coroutine objects (tasks) but do not await them yet
            tasks = [
                extract_single_document(
                    pdf_path=task["pdf_path"],
                    doc_type=task["doc_type"],
                    page_range=task["page_range"],
                    doc_index=task["doc_index"]
                )
                for task in batch
            ]
            
            # 2. Fire all tasks in this batch AT THE SAME TIME using gather
            # return_exceptions=True prevents one failure from crashing the batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            for res in batch_results:
                if isinstance(res, Exception):
                    logger.error(f"Batch task failed: {res}")
                    all_results.append(res)
                else:
                    all_results.append(res)
            
            if batch_idx < num_batches - 1:
                logger.debug(f"Job {job_id}: Waiting {inter_batch_delay}s before next batch...")
                await asyncio.sleep(inter_batch_delay)
        
        extraction_time = time.time() - extraction_start
        logger.info(f"Job {job_id}: All {len(all_results)} extractions completed in {extraction_time:.2f}s")
        
        # Process results
        final_results = []
        total_tokens = 0
        total_cost = 0.0
        
        for result in all_results:
            if isinstance(result, Exception):
                logger.error(f"Extraction exception: {result}")
                continue
            if isinstance(result, ExtractionResult):
                final_results.append(result)
                if result.token_usage:
                    total_tokens += result.token_usage.get("total_token_count", 0)
                total_cost += result.cost_usd or 0.0
        
        # Finalize
        JOBS[job_id]["results"] = [r.dict() for r in final_results]
        JOBS[job_id]["total_tokens"] = total_tokens
        JOBS[job_id]["total_cost_usd"] = round(total_cost, 6)
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["completed_at"] = datetime.now()
        JOBS[job_id]["processing_time_seconds"] = round(time.time() - pipeline_start, 2)
        
        # ===== DASHBOARD: COMPLETION =====
        success_count = sum(1 for r in final_results if r.error is None)
        fail_count = sum(1 for r in final_results if r.error is not None)
        
        print("\n" + "="*60)
        print(f"✅ JOB COMPLETED (Job: {job_id[:8]}...)")
        print("="*60)
        print(f"⏱️  Total time: {JOBS[job_id]['processing_time_seconds']}s")
        print(f"📄 Documents extracted: {success_count}/{len(final_results)}")
        if fail_count > 0:
            print(f"❌ Failed: {fail_count}")
            for r in final_results:
                if r.error:
                    print(f"   - {r.document_type} (pages {r.page_range}): {r.error[:50]}...")
        print(f"💰 Total cost: ${total_cost:.4f}")
        print(f"🔢 Total tokens: {total_tokens}")
        
        # Rate limiter stats
        try:
            stats = get_rate_limit_stats()
            print("\n🛡️ Rate Limiter Stats:")
            for model, st in stats.items():
                print(f"   {model}: {st['current_window']}/{st['limit_rpm']} RPM, waited {st['total_wait_seconds']}s total")
        except:
            pass
        
        print("="*60 + "\n")
        
        logger.info(f"Job {job_id}: COMPLETED in {JOBS[job_id]['processing_time_seconds']}s")
        
    except Exception as e:
        logger.error(f"Job {job_id} Failed: {e}")
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)

# ==========================================
# 7. FASTAPI APPLICATION
# ==========================================
app = FastAPI(title="Logistics Extraction API (Vendor Routing + Parallel)", version="3.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """
    Initialize bulletproof rate limiter on application startup.
    
    Rate Limits (from Google AI Studio dashboard):
    - gemini-2.5-flash: 1000 RPM, 1M TPM, 10K RPD
    - gemini-2.5-pro: 150 RPM, 2M TPM, 10K RPD
    
    With 70% safety margin:
    - Flash: 700 effective RPM
    - Pro: 105 effective RPM (~1.75 req/sec)
    """
    initialize_rate_limiter(
        model_limits={
            "gemini-2.5-flash": 1000,  # Classification model
            "gemini-2.5-pro": 150,     # Extraction model - THIS IS THE BOTTLENECK
        },
        safety_margin=0.7,             # 70% of limit (more conservative)
        min_request_gap_ms=500,        # Minimum 500ms between requests to same model
        max_concurrent_per_model=2     # Max 2 concurrent requests per model
    )
    logger.info("✅ Bulletproof rate limiter initialized for 40+ split handling")

JOBS = {}
UPLOAD_DIR = "uploads"
SPLIT_DIR = "splits"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

@app.post("/api/v1/process-document", response_model=JobStatusResponse)
async def process_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF allowed.")

    job_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}.pdf")
    
    try:
        content = await file.read()
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
    
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now(),
        "file_name": file.filename,
        "model_used": CONFIG["EXTRACTOR_MODEL"]
    }
    
    await run_pipeline_optimized(job_id, file_path, CONFIG["EXTRACTOR_MODEL"])
    
    return JobStatusResponse(**JOBS[job_id])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8988, timeout_keep_alive=300)