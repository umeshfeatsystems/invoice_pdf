"""
Optimized Logistics Document Processing Pipeline
=================================================
Key Optimizations:
1. Parallel extraction using asyncio.gather()
2. Reusable GenAI model instances
3. Async file I/O with aiofiles
4. Semaphore-based concurrency control
5. Exponential backoff retry with quota detection
6. Batch processing with inter-batch delays
7. Rate limiting to stay within API quotas
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
    return logging.getLogger("logistics_optimized")

logger = setup_logging()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables.")
else:
    genai.configure(api_key=API_KEY)

CONFIG = {
    "CLASSIFIER_MODEL": "gemini-2.5-flash",
    "EXTRACTOR_MODEL": "gemini-2.5-flash",  # Using flash for speed
    "API_TIMEOUT": 120,  # Reduced timeout for faster failure detection
    
    # Rate Limiting & Batch Processing
    "MAX_CONCURRENT_EXTRACTIONS": 3,   # Conservative: 3 parallel calls at a time
    "BATCH_SIZE": 3,                    # Process in batches of 3
    "INTER_BATCH_DELAY_SECONDS": 1.0,   # Wait 1s between batches
    "MIN_DELAY_BETWEEN_CALLS": 0.2,     # 200ms minimum between API calls
    
    # Retry Configuration
    "MAX_RETRIES": 6,
    "INITIAL_RETRY_DELAY": 2.0,         # Start with 2 second delay
    "MAX_RETRY_DELAY": 60.0,            # Cap at 60 seconds
    "RETRY_MULTIPLIER": 2.0,            # Exponential backoff multiplier
}

PRICING = {
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "default": {"input": 0.10, "output": 0.40}
}

# Known quota/rate limit error patterns
QUOTA_ERROR_PATTERNS = [
    "quota", "rate limit", "resource exhausted", "429",
    "too many requests", "exceeded", "limit exceeded"
]

# Thread pool for sync operations
thread_pool = ThreadPoolExecutor(max_workers=4)

# Semaphore for API rate limiting (initialized lazily)
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

def get_pdf_page_count(file_content: bytes) -> int:
    try:
        reader = PdfReader(io.BytesIO(file_content))
        return len(reader.pages)
    except Exception:
        return 0

def split_pdf_sync(original_pdf_path: str, ranges: List[List[int]], output_dir: str, prefix: str) -> List[str]:
    """Synchronous PDF splitting - run in thread pool."""
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
    """Async wrapper for PDF splitting using thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        thread_pool, 
        split_pdf_sync, 
        original_pdf_path, ranges, output_dir, prefix
    )


def is_quota_error(error: Exception) -> bool:
    """Check if an error is related to API quota/rate limits."""
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
    """
    Execute an async function with exponential backoff retry.
    
    Features:
    - Exponential delay increase on each retry
    - Jitter to prevent thundering herd
    - Special handling for quota errors (longer delays)
    - Configurable via CONFIG dict
    
    Returns the result or raises the last exception.
    """
    max_retries = max_retries or CONFIG["MAX_RETRIES"]
    initial_delay = initial_delay or CONFIG["INITIAL_RETRY_DELAY"]
    max_delay = max_delay or CONFIG["MAX_RETRY_DELAY"]
    multiplier = multiplier or CONFIG["RETRY_MULTIPLIER"]
    
    last_error = None
    delay = initial_delay
    
    for attempt in range(max_retries + 1):
        try:
            # Add minimum delay between calls to be nice to API
            if attempt > 0:
                await asyncio.sleep(CONFIG["MIN_DELAY_BETWEEN_CALLS"])
            
            return await async_func()
            
        except Exception as e:
            last_error = e
            
            # Don't retry on last attempt
            if attempt >= max_retries:
                break
            
            # Check if it's a quota error - use longer delays
            if is_quota_error(e):
                delay = min(delay * 2, max_delay)  # Double delay for quota errors
                logger.warning(
                    f"[{operation_name}] Quota error detected (attempt {attempt + 1}/{max_retries + 1}). "
                    f"Waiting {delay:.1f}s before retry..."
                )
            else:
                logger.warning(
                    f"[{operation_name}] Error (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
            
            # Add jitter (±20%) to prevent thundering herd
            jitter = delay * 0.2 * (random.random() * 2 - 1)
            await asyncio.sleep(delay + jitter)
            
            # Exponential increase for next attempt
            delay = min(delay * multiplier, max_delay)
    
    raise last_error

# ==========================================
# 3. PYDANTIC SCHEMAS (Updated field names)
# ==========================================
class InvoiceItem(BaseModel):
    """Line item from an invoice - field names match production data format."""
    item_no: Optional[str] = None
    item_description: Optional[str] = None
    item_part_no: Optional[str] = None
    item_po_no: Optional[str] = None
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
    """Invoice document structure - field names match production data format."""
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
    extraction_time_seconds: Optional[float] = None  # NEW: per-document timing


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
    # NEW: Performance metrics
    classification_time_seconds: Optional[float] = None
    extraction_parallelism: Optional[int] = None

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
# 5. OPTIMIZED AI SERVICE FUNCTIONS
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

    # Upload file (this is I/O bound)
    pdf_file = genai.upload_file(pdf_path, mime_type="application/pdf")
    model = get_classifier_model()
    config = get_generation_config(response_schema=DocumentClassification)
    
    try:
        response = await model.generate_content_async(
            [CLASSIFICATION_PROMPT, pdf_file],
            generation_config=config,
            request_options={"timeout": CONFIG["API_TIMEOUT"]}
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
    Extract a single document with:
    - Semaphore-based rate limiting
    - Exponential backoff retry
    - Quota error detection
    
    Designed to be called in parallel with asyncio.gather().
    """
    start_time = time.time()
    
    async with get_api_semaphore():  # Rate limiting via semaphore
        try:
            # Define the extraction logic as an async function for retry
            async def do_extraction():
                pdf_file = genai.upload_file(pdf_path, mime_type="application/pdf")
                model = get_extractor_model()
                
                if doc_type == "invoice":
                    prompt = INVOICE_PROMPT
                    schema = Invoice
                else:
                    prompt = AWB_PROMPT
                    schema = AirwayBill
                
                config = get_generation_config(response_schema=schema)
                
                response = await model.generate_content_async(
                    [prompt, pdf_file],
                    generation_config=config,
                    request_options={"timeout": CONFIG["API_TIMEOUT"]}
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
            
            # Post-processing
            if doc_type == "invoice":
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


def post_process_invoice(data: Dict[str, Any]) -> Dict[str, Any]:
    """Post-process invoice data."""
    toi = data.get("invoice_toi", "") or ""
    if toi and "FCA" in str(toi).upper(): 
        data["invoice_toi"] = "FOB"
    if "items" in data and isinstance(data["items"], list):
        for item in data["items"]:
            desc = item.get("item_description", "") or ""
            part = item.get("item_part_no", "") or ""
            if part and part not in desc:
                item["item_description"] = f"{desc} {part}".strip()
    return data


def post_process_airway_bill(data: Dict[str, Any]) -> Dict[str, Any]:
    """Post-process AWB data."""
    def clean_awb(value):
        if not value: return None
        return "".join(filter(str.isalnum, str(value)))
    if "master_awb_no" in data: 
        data["master_awb_no"] = clean_awb(data["master_awb_no"])
    if "house_awb_no" in data: 
        data["house_awb_no"] = clean_awb(data["house_awb_no"])
    return data


# ==========================================
# 6. OPTIMIZED PIPELINE
# ==========================================
async def run_pipeline_optimized(job_id: str, file_path: str, model_name: str):
    """
    Optimized pipeline with PARALLEL extraction.
    
    Performance improvements:
    - Classification runs first (single call)
    - All document extractions run in PARALLEL using asyncio.gather()
    - Semaphore limits concurrent API calls to avoid rate limits
    """
    try:
        JOBS[job_id]["status"] = "classifying"
        pipeline_start = time.time()
        
        # Step 1: Classification
        logger.info(f"Job {job_id}: Classifying...")
        classification, class_time = await classify_documents_optimized(file_path)
        JOBS[job_id]["classification"] = classification.dict()
        JOBS[job_id]["classification_time_seconds"] = round(class_time, 2)
        
        # Step 2: Split PDF (async in thread pool)
        JOBS[job_id]["status"] = "splitting"
        split_tasks = []
        for r in classification.invoices: 
            split_tasks.append({"type": "invoice", "range": r})
        for r in classification.airway_bills: 
            split_tasks.append({"type": "airway_bill", "range": r})
        
        ranges = [t["range"] for t in split_tasks]
        if not ranges:
            split_tasks = [{"type": "invoice", "range": [1, 1]}]
            split_paths = [file_path]
        else:
            split_paths = await split_pdf_async(
                file_path, ranges, 
                os.path.join(SPLIT_DIR, job_id), job_id
            )
        
        # Step 3: BATCH EXTRACTION with rate limiting 🚀
        JOBS[job_id]["status"] = "extracting"
        JOBS[job_id]["extraction_parallelism"] = min(len(split_paths), CONFIG["MAX_CONCURRENT_EXTRACTIONS"])
        
        extraction_start = time.time()
        batch_size = CONFIG["BATCH_SIZE"]
        inter_batch_delay = CONFIG["INTER_BATCH_DELAY_SECONDS"]
        
        # Build list of extraction tasks
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
        
        logger.info(f"Job {job_id}: Processing {len(extraction_tasks)} documents in {num_batches} batches (batch_size={batch_size})")
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(extraction_tasks))
            batch = extraction_tasks[batch_start:batch_end]
            
            logger.info(f"Job {job_id}: Starting batch {batch_idx + 1}/{num_batches} ({len(batch)} docs)")
            
            # Create coroutines for this batch
            batch_coroutines = [
                extract_single_document(
                    pdf_path=task["pdf_path"],
                    doc_type=task["doc_type"],
                    page_range=task["page_range"],
                    doc_index=task["doc_index"]
                )
                for task in batch
            ]
            
            # Run batch in parallel (limited by semaphore)
            batch_results = await asyncio.gather(*batch_coroutines, return_exceptions=True)
            all_results.extend(batch_results)
            
            # Add delay between batches to avoid hitting rate limits
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
        
        logger.info(f"Job {job_id}: COMPLETED in {JOBS[job_id]['processing_time_seconds']}s")
        
    except Exception as e:
        logger.error(f"Job {job_id} Failed: {e}")
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)


# ==========================================
# 7. FASTAPI APPLICATION
# ==========================================
app = FastAPI(
    title="Logistics Document Extraction API (Optimized)",
    description="High-performance document extraction with parallel processing",
    version="2.0.0"
)

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


@app.get("/health")
async def health_check():
    """Quick health check endpoint."""
    return {"status": "healthy", "model": CONFIG["EXTRACTOR_MODEL"]}


@app.post("/api/v1/process-document", response_model=JobStatusResponse)
async def process_document(file: UploadFile = File(...)):
    """
    Process PDF document with optimized parallel extraction.
    
    Performance Features:
    - Parallel document extraction
    - Semaphore-based rate limiting
    - Per-document timing metrics
    """
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
    
    # Initialize Job
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now(),
        "file_name": file.filename,
        "model_used": CONFIG["EXTRACTOR_MODEL"]
    }
    
    # Run optimized pipeline
    await run_pipeline_optimized(job_id, file_path, CONFIG["EXTRACTOR_MODEL"])
    
    return JobStatusResponse(**JOBS[job_id])


@app.post("/api/v1/process-document-async")
async def process_document_async(file: UploadFile = File(...)):
    """
    Non-blocking endpoint that returns immediately with job_id.
    Use /api/v1/job/{job_id} to check status.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF allowed.")

    job_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}.pdf")
    
    content = await file.read()
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)
    
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now(),
        "file_name": file.filename,
        "model_used": CONFIG["EXTRACTOR_MODEL"]
    }
    
    # Fire and forget - runs in background
    asyncio.create_task(run_pipeline_optimized(job_id, file_path, CONFIG["EXTRACTOR_MODEL"]))
    
    return {"job_id": job_id, "status": "queued", "message": "Processing started. Poll /api/v1/job/{job_id} for status."}


@app.get("/api/v1/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Check the status of a processing job."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**JOBS[job_id])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=300)
