"""
Pydantic Schemas for Document Extraction
=========================================
These schemas define the structure of extracted data.
Field names match the prompt_config.py configuration.
"""

from typing import List, Optional, Any
from pydantic import BaseModel, Field

# --- Nested Item Models ---

class InvoiceItem(BaseModel):
    """Line item from an invoice."""
    item_no: Optional[str] = None
    item_description: Optional[str] = None
    item_part_no: Optional[str] = None
    item_po_no: Optional[str] = None
    
    # NEW: Date field
    item_date: Optional[str] = Field(None, description="Date specific to the line item (YYYY-MM-DD)")
    
    # Tariff/Customs codes
    hsn_code: Optional[str] = Field(None, description="Harmonized System Code")
    item_cth: Optional[str] = Field(None, description="Customs Tariff Heading")
    item_ritc: Optional[str] = Field(None, description="Regional Import Tariff Code")
    item_ceth: Optional[str] = Field(None, description="Central Excise Tariff Heading")
    
    # Quantity and pricing
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


# --- Document Models ---

class Invoice(BaseModel):
    """Invoice document structure."""
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    
    # Parties
    seller_name: Optional[str] = Field(None, description="Supplier Name")
    seller_address: Optional[str] = None
    buyer_name: Optional[str] = None
    buyer_address: Optional[str] = None
    
    # Financial
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


# --- API Response Models ---

class DocumentClassification(BaseModel):
    """
    Classification result showing page ranges for each document type.
    Now supports specific vendor routing.
    """
    # Standard documents
    invoices: List[List[int]] = Field(default_factory=list, description="Standard/Global Invoices")
    
    # Specific Vendor Invoices
    abb_invoices: List[List[int]] = Field(default_factory=list, description="ABB or Epiroc Invoices")
    crown_invoices: List[List[int]] = Field(default_factory=list, description="Crown Worldwide Invoices")
    
    # Logistics
    airway_bills: List[List[int]] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Result of extracting data from a single document."""
    document_type: str  # "invoice", "abb_invoice", "crown_invoice", "airway_bill"
    page_range: List[int]
    data: Optional[dict] = None 
    error: Optional[str] = None
    token_usage: Optional[dict] = None
    cost_usd: Optional[float] = 0.0
    extraction_time_seconds: Optional[float] = None


class JobStatusResponse(BaseModel):
    """Full status response for a processing job."""
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