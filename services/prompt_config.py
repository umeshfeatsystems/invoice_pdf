"""
Standardized JSON-based Prompt Configuration System
====================================================
Updated for Specific Vendor Routing:
1. Commin (Murata / SIN-...) - UPDATED
2. Type 1 (0100935473...)
3. Type 2 (KM_558...)
4. Crown
5. ABB / Epiroc
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import copy

class FieldType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    DATE = "date"
    CURRENCY = "currency"
    LIST = "list"

@dataclass
class FieldConfig:
    name: str                           
    display_name: str                   
    field_type: FieldType               
    description: str                    
    extraction_guidelines: List[str]    
    aliases: List[str] = None           
    required: bool = False              
    default: Any = None                 
    validation_rules: List[str] = None  
    
    def __post_init__(self):
        if self.aliases is None: self.aliases = []
        if self.validation_rules is None: self.validation_rules = []

# =============================================================================
# BASE INVOICE FIELDS (GLOBAL)
# =============================================================================
INVOICE_FIELDS: Dict[str, FieldConfig] = {
    "invoice_number": FieldConfig(name="invoice_number", display_name="Invoice Number", field_type=FieldType.STRING, description="Unique invoice identifier", extraction_guidelines=["Look for 'Invoice No', 'Invoice #', 'Inv No'"], required=True),
    "invoice_date": FieldConfig(name="invoice_date", display_name="Invoice Date", field_type=FieldType.DATE, description="Date the invoice was issued", extraction_guidelines=["ISO format: YYYY-MM-DD"], required=True),
    "seller_name": FieldConfig(name="seller_name", display_name="Seller Name", field_type=FieldType.STRING, description="Name of the seller/supplier company", extraction_guidelines=["Company name at top"], required=True),
    "seller_address": FieldConfig(name="seller_address", display_name="Seller Address", field_type=FieldType.STRING, description="Complete address of the seller", extraction_guidelines=["Below seller name"]),
    "buyer_name": FieldConfig(name="buyer_name", display_name="Buyer Name", field_type=FieldType.STRING, description="Name of the buyer/importer company", extraction_guidelines=["Look for 'Buyer', 'Consignee', 'Bill To'"], required=True),
    "buyer_address": FieldConfig(name="buyer_address", display_name="Buyer Address", field_type=FieldType.STRING, description="Complete address of the buyer", extraction_guidelines=["Below buyer name"]),
    "invoice_currency": FieldConfig(name="invoice_currency", display_name="Currency", field_type=FieldType.CURRENCY, description="Currency code", extraction_guidelines=["ISO code (USD, EUR)"]),
    "total_amount": FieldConfig(name="total_amount", display_name="Total Amount", field_type=FieldType.NUMBER, description="Grand total", extraction_guidelines=["Numeric only"], required=True),
    "invoice_toi": FieldConfig(name="invoice_toi", display_name="Terms of Invoice", field_type=FieldType.STRING, description="IncoTerms", extraction_guidelines=["EXW, FCA, FOB, CIF, etc."]),
    "invoice_exchange_rate": FieldConfig(name="invoice_exchange_rate", display_name="Exchange Rate", field_type=FieldType.NUMBER, description="Exchange rate", extraction_guidelines=["Numeric"]),
    "invoice_po_date": FieldConfig(name="invoice_po_date", display_name="PO Date", field_type=FieldType.DATE, description="PO Date", extraction_guidelines=["ISO format"]),
}

INVOICE_ITEM_FIELDS: Dict[str, FieldConfig] = {
    "item_no": FieldConfig(name="item_no", display_name="Item Number", field_type=FieldType.STRING, description="Serial number", default="1", extraction_guidelines=["HARDCODED: ALWAYS '1'"]),
    "item_description": FieldConfig(name="item_description", display_name="Description", field_type=FieldType.STRING, description="Description", required=True, extraction_guidelines=["Combine Description + Part Number"]),
    "item_part_no": FieldConfig(name="item_part_no", display_name="Part Number", field_type=FieldType.STRING, description="Part number/SKU", extraction_guidelines=["Look for Part No, P/N, SKU"]),
    "item_po_no": FieldConfig(name="item_po_no", display_name="Item PO Number", field_type=FieldType.STRING, description="PO Reference", extraction_guidelines=["Look for 'Your reference', 'Customer PO', 'Order No'"]),
    "item_date": FieldConfig(name="item_date", display_name="Item Date", field_type=FieldType.DATE, description="Item specific date", extraction_guidelines=["ISO format"]),
    "hsn_code": FieldConfig(name="hsn_code", display_name="HSN Code", field_type=FieldType.STRING, description="HS Code", extraction_guidelines=["4-10 digits"]),
    "item_cth": FieldConfig(name="item_cth", display_name="CTH", field_type=FieldType.STRING, description="Customs Tariff Heading", extraction_guidelines=["Same as HSN"]),
    "item_ritc": FieldConfig(name="item_ritc", display_name="RITC", field_type=FieldType.STRING, description="RITC Code", extraction_guidelines=["Same as HSN"]),
    "item_ceth": FieldConfig(name="item_ceth", display_name="CETH", field_type=FieldType.STRING, description="CETH Code", extraction_guidelines=["Same as HSN"]),
    "item_quantity": FieldConfig(name="item_quantity", display_name="Quantity", field_type=FieldType.NUMBER, description="Quantity", required=True, extraction_guidelines=["Numeric"]),
    "item_uom": FieldConfig(name="item_uom", display_name="UOM", field_type=FieldType.STRING, description="Unit of Measure", extraction_guidelines=["PCS, KG, etc."]),
    "item_unit_price": FieldConfig(name="item_unit_price", display_name="Unit Price", field_type=FieldType.NUMBER, description="Price per unit", extraction_guidelines=["Numeric"]),
    "item_amount": FieldConfig(name="item_amount", display_name="Amount", field_type=FieldType.NUMBER, description="Line total", extraction_guidelines=["Numeric"]),
    "item_origin_country": FieldConfig(name="item_origin_country", display_name="Origin", field_type=FieldType.STRING, description="Country of Origin", extraction_guidelines=["COO, Made In"]),
    "item_mfg_name": FieldConfig(name="item_mfg_name", display_name="Manufacturer", field_type=FieldType.STRING, description="Manufacturer Name", extraction_guidelines=["Name"]),
    "item_mfg_addr": FieldConfig(name="item_mfg_addr", display_name="Mfg Address", field_type=FieldType.STRING, description="Manufacturer Address", extraction_guidelines=["Address"]),
    "item_mfg_country": FieldConfig(name="item_mfg_country", display_name="Mfg Country", field_type=FieldType.STRING, description="Manufacturer Country", extraction_guidelines=["Country"]),
}

AIRWAY_BILL_FIELDS: Dict[str, FieldConfig] = {
    "master_awb_no": FieldConfig(name="master_awb_no", display_name="Master AWB", field_type=FieldType.STRING, description="MAWB", required=True, extraction_guidelines=["11 Digits"]),
    "house_awb_no": FieldConfig(name="house_awb_no", display_name="House AWB", field_type=FieldType.STRING, description="HAWB", extraction_guidelines=["Alphanumeric"]),
    "shipper_name": FieldConfig(name="shipper_name", display_name="Shipper", field_type=FieldType.STRING, description="Shipper Name", extraction_guidelines=["Sender"]),
    "pkg_in_qty": FieldConfig(name="pkg_in_qty", display_name="Pkg Qty", field_type=FieldType.NUMBER, description="Package Count", required=True, extraction_guidelines=["Numeric"]),
    "gross_weight_kg": FieldConfig(name="gross_weight_kg", display_name="Gross Weight", field_type=FieldType.NUMBER, description="Gross Weight", required=True, extraction_guidelines=["Numeric"]),
    "chargeable_weight_kg": FieldConfig(name="chargeable_weight_kg", display_name="Chrg Weight", field_type=FieldType.NUMBER, description="Chargeable Weight", required=True, extraction_guidelines=["Numeric"]),
    "total_frieght": FieldConfig(name="total_frieght", display_name="Freight", field_type=FieldType.NUMBER, description="Total Freight", extraction_guidelines=["Numeric"]),
    "frieght_currency": FieldConfig(name="frieght_currency", display_name="Freight Currency", field_type=FieldType.CURRENCY, description="Currency", extraction_guidelines=["ISO Code"]),
}

# =============================================================================
# 5-WAY VENDOR SPECIFIC OVERRIDES
# =============================================================================

# Helper to clone and update fields
def create_custom_fields(po_override=None, part_override=None):
    fields = copy.deepcopy(INVOICE_ITEM_FIELDS)
    if po_override: fields["item_po_no"] = po_override
    if part_override: fields["item_part_no"] = part_override
    return fields

# --- 1. COMMIN (SIN-10076768 / MURATA) ---
COMMIN_PART = FieldConfig(
    name="item_part_no", 
    display_name="Part Number (Commin)", 
    field_type=FieldType.STRING, 
    description="Customer Part Number",
    extraction_guidelines=[
        "**STRICT TARGETING**: Extract value ONLY from column 'CUSTOMER PART NUMBER'.",
        "**NEGATIVE RULE**: Do NOT extract 'ARTICLE' or 'CUSTOMER REFERENCE PART #'.",
        "Example: If row has '114304360' and 'GRM31...', extract '114304360'."
    ]
)
COMMIN_FIELDS = create_custom_fields(part_override=COMMIN_PART)

# --- 2. TYPE 1 (0100935473) ---
TYPE1_PO = FieldConfig(
    name="item_po_no", display_name="PO Number (Type 1)", field_type=FieldType.STRING, description="Order No",
    extraction_guidelines=["**STRICT**: Extract value 'ORDER NO:'"]
)
TYPE1_PART = FieldConfig(
    name="item_part_no", display_name="Part Number (Type 1)", field_type=FieldType.STRING, description="Article Number",
    extraction_guidelines=[
        "**STRICT**: Extract from 'Article number' column (e.g. 8940162882)",
        "**NEGATIVE RULE**: Do NOT extract text starting with 'Pack:' or 'Pack: 103...'",
        "Ignore 'NW 104,000'"
    ]
)
TYPE1_FIELDS = create_custom_fields(po_override=TYPE1_PO, part_override=TYPE1_PART)

# --- 3. TYPE 2 (KM_558...) ---
TYPE2_PO = FieldConfig(
    name="item_po_no", display_name="PO Number (Type 2)", field_type=FieldType.STRING, description="Order No",
    extraction_guidelines=["**STRICT**: Extract 'ORDER NO'"]
)
TYPE2_PART = FieldConfig(
    name="item_part_no", display_name="Part Number (Type 2)", field_type=FieldType.STRING, description="Article Number",
    extraction_guidelines=[
        "**STRICT**: Extract from 'Article number' column (e.g. 4081041590)",
        "**NEGATIVE RULE**: Do NOT extract 'Pack number'"
    ]
)
TYPE2_FIELDS = create_custom_fields(po_override=TYPE2_PO, part_override=TYPE2_PART)

# --- 4. CROWN ---
CROWN_PO = FieldConfig(
    name="item_po_no", display_name="PO Number (Crown)", field_type=FieldType.STRING, description="Customer P.O.",
    extraction_guidelines=[
        "**STRICT**: Extract 'Customer P.O.'",
        "**NEGATIVE**: Ignore 'Order Number', 'Shipping number', 'Packing Slip'"
    ]
)
CROWN_FIELDS = create_custom_fields(po_override=CROWN_PO)

# --- 5. ABB ---
ABB_PO = FieldConfig(
    name="item_po_no", display_name="PO Number (ABB)", field_type=FieldType.STRING, description="Ref/Order No",
    extraction_guidelines=["**STRICT**: Extract 'Your reference', 'Your orderNo.', 'Our order no'"]
)
ABB_FIELDS = create_custom_fields(po_override=ABB_PO)


# =============================================================================
# PROMPT GENERATOR
# =============================================================================
def generate_field_prompt_section(fields: Dict[str, FieldConfig], section_name: str = "FIELDS") -> str:
    lines = [f"### {section_name} TO EXTRACT:"]
    for _, config in fields.items():
        req = " [REQUIRED]" if config.required else ""
        lines.append(f"\n- **{config.name}** ({config.display_name}){req}")
        lines.append(f"  - Description: {config.description}")
        if config.extraction_guidelines:
            lines.append("  - Guidelines:")
            for g in config.extraction_guidelines: lines.append(f"    • {g}")
    return "\n".join(lines)

def generate_extraction_prompt(doc_type, fields, item_fields, custom_instructions=None):
    parts = [f"You are a forensic Customs Data Extractor. Extract data from this {doc_type}.\n"]
    if custom_instructions:
        parts.append("### CRITICAL BUSINESS RULES:")
        for rule in custom_instructions: parts.append(f"- {rule}")
    parts.append("\n")
    parts.append(generate_field_prompt_section(fields, "DOCUMENT FIELDS"))
    if item_fields:
        parts.append("\n")
        parts.append(generate_field_prompt_section(item_fields, "LINE ITEM FIELDS"))
        parts.append("\n### LINE ITEMS INSTRUCTIONS:")
        parts.append("- Extract ALL line items from the table")
        parts.append("- **item_no**: HARDCODED RULE: ALWAYS set to '1'")
        parts.append("- **item_part_no / item_po_no**: FOLLOW STRICT NEGATIVE RULES.")
        
    parts.append("\n### OUTPUT:")
    parts.append("Return valid JSON strictly matching the provided schema. Use null for missing fields.")
    return "\n".join(parts)

# =============================================================================
# FINAL PROMPTS
# =============================================================================

CLASSIFICATION_PROMPT = """
You are a Senior Document Router. Classify the PDF into one of 6 categories.

### 1. COMMIN INVOICE ("commin_invoices")
   - **Identifiers**: "SIN-10076768", "COMMIN", "MURATA ELECTRONICS", "SALES CARD".
   - **Key Column**: "CUSTOMER PART NUMBER".

### 2. TYPE 1 INVOICE ("type1_invoices")
   - **Identifiers**: "0100935473", "Article number" column, "Pack:" text in column.
   - Ref: "0100935473_0302000031.pdf"

### 3. TYPE 2 INVOICE ("type2_invoices")
   - **Identifiers**: "KM_558", "Article number" (but NO "Pack:" text).
   - Ref: "KM_558e25070116141.pdf"

### 4. CROWN INVOICE ("crown_invoices")
   - **Identifiers**: "Crown Worldwide", "Customer P.O."
   - Ref: "CROWN INVOICE01.pdf"

### 5. ABB INVOICE ("abb_invoices")
   - **Identifiers**: "ABB", "Epiroc", "FJ1039 INV"
   - Ref: "FJ1039 INV"

### 6. AIRWAY BILL ("airway_bills")
   - **Identifiers**: "Master AWB", "MAWB", "HAWB"

### 7. GLOBAL INVOICE ("invoices")
   - Any invoice not matching the above 5 specific types.

### OUTPUT JSON:
{
  "commin_invoices": [[start, end]],
  "type1_invoices": [[start, end]],
  "type2_invoices": [[start, end]],
  "crown_invoices": [[start, end]],
  "abb_invoices": [[start, end]],
  "invoices": [[start, end]],
  "airway_bills": [[start, end]]
}
"""

COMMIN_PROMPT = generate_extraction_prompt(
    "Commin Invoice (Murata)", 
    INVOICE_FIELDS, 
    COMMIN_FIELDS, 
    [
        "**STRICT**: item_part_no must come from 'CUSTOMER PART NUMBER' column.", 
        "**IGNORE**: 'CUSTOMER REFERENCE PART' or 'ARTICLE' for item_part_no.",
        "**MANUFACTURER**: Extract 'item_mfg_name' and 'item_mfg_addr' from the SELLER/HEADER block (e.g. Murata Electronics Singapore).",
        "**ORIGIN**: Extract 'item_origin_country' from 'COUNTRY OF ORIGIN' column (e.g. JAPAN)."
    ]
)

TYPE1_PROMPT = generate_extraction_prompt(
    "Type 1 Invoice", 
    INVOICE_FIELDS, 
    TYPE1_FIELDS, 
    ["**STRICT**: item_po_no = 'ORDER NO'", "**STRICT**: item_part_no = 'Article number'", "**IGNORE**: 'Pack:' or 'NW' in Article column"]
)

TYPE2_PROMPT = generate_extraction_prompt(
    "Type 2 Invoice", 
    INVOICE_FIELDS, 
    TYPE2_FIELDS, 
    ["**STRICT**: item_po_no = 'ORDER NO'", "**STRICT**: item_part_no = 'Article number'", "**IGNORE**: 'Pack number'"]
)

CROWN_PROMPT = generate_extraction_prompt(
    "Crown Invoice", 
    INVOICE_FIELDS, 
    CROWN_FIELDS, 
    ["**STRICT**: item_po_no = 'Customer P.O.'", "**IGNORE**: 'Order Number', 'Shipping number', 'Packing Slip'"]
)

ABB_PROMPT = generate_extraction_prompt(
    "ABB Invoice", 
    INVOICE_FIELDS, 
    ABB_FIELDS, 
    ["**STRICT**: item_po_no = 'Your reference' OR 'Your orderNo.'"]
)

INVOICE_PROMPT = generate_extraction_prompt(
    "Commercial Invoice", 
    INVOICE_FIELDS, 
    INVOICE_ITEM_FIELDS, 
    ["General extraction rules apply"]
)

AWB_PROMPT = generate_extraction_prompt(
    "Airway Bill", 
    AIRWAY_BILL_FIELDS, 
    None, 
    ["Extract MAWB (11 digits) and HAWB"]
)