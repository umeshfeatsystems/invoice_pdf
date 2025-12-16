"""
Standardized JSON-based Prompt Configuration System
====================================================
This module defines extraction fields and their guidelines in a structured JSON format.
Adding/removing fields is as simple as editing the FIELD_CONFIG dictionaries.

Field naming follows the convention from production data:
- Document fields: invoice_* prefix
- Item fields: item_* prefix
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class FieldType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    DATE = "date"
    CURRENCY = "currency"
    LIST = "list"


@dataclass
class FieldConfig:
    """Configuration for a single extraction field."""
    name: str                           # JSON key name (e.g., "seller_name")
    display_name: str                   # Human readable name (e.g., "Seller Name")
    field_type: FieldType               # Data type
    description: str                    # What this field represents
    extraction_guidelines: List[str]    # How to extract this field
    aliases: List[str] = None           # Alternative names to look for in document
    required: bool = False              # Is this field mandatory
    default: Any = None                 # Default value if not found
    validation_rules: List[str] = None  # Validation instructions
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.validation_rules is None:
            self.validation_rules = []


# =============================================================================
# INVOICE FIELD CONFIGURATION
# =============================================================================
INVOICE_FIELDS: Dict[str, FieldConfig] = {
    # --- Header Fields ---
    "invoice_number": FieldConfig(
        name="invoice_number",
        display_name="Invoice Number",
        field_type=FieldType.STRING,
        description="Unique invoice identifier",
        extraction_guidelines=[
            "Look for 'Invoice No', 'Invoice #', 'Inv No', 'Bill No'",
            "Usually alphanumeric, may contain dashes or slashes",
            "Found typically at the top of the invoice"
        ],
        aliases=["Invoice No", "Invoice #", "Inv No", "Bill No", "Invoice Number"],
        required=True
    ),
    "invoice_date": FieldConfig(
        name="invoice_date",
        display_name="Invoice Date",
        field_type=FieldType.DATE,
        description="Date the invoice was issued",
        extraction_guidelines=[
            "Look for 'Invoice Date', 'Date', 'Dated'",
            "Return in ISO format: YYYY-MM-DD",
            "If multiple dates exist, prefer the one labeled 'Invoice Date'"
        ],
        aliases=["Invoice Date", "Date", "Dated", "Inv Date"],
        required=True
    ),
    "seller_name": FieldConfig(
        name="seller_name",
        display_name="Seller Name",
        field_type=FieldType.STRING,
        description="Name of the seller/supplier company",
        extraction_guidelines=[
            "Look for 'Seller', 'Supplier', 'Vendor', 'Exporter', 'From'",
            "Usually the company name at the top of the invoice",
            "May be in letterhead or header section"
        ],
        aliases=["Seller", "Supplier", "Vendor", "Exporter", "From", "Shipper"],
        required=True
    ),
    "seller_address": FieldConfig(
        name="seller_address",
        display_name="Seller Address",
        field_type=FieldType.STRING,
        description="Complete address of the seller",
        extraction_guidelines=[
            "Extract full address including street, city, state, country, postal code",
            "Usually directly below seller name",
            "Combine multiple address lines into single string"
        ],
        aliases=["Seller Address", "Supplier Address", "From Address"]
    ),
    "buyer_name": FieldConfig(
        name="buyer_name",
        display_name="Buyer Name",
        field_type=FieldType.STRING,
        description="Name of the buyer/importer company",
        extraction_guidelines=[
            "Look for 'Buyer', 'Consignee', 'Bill To', 'Ship To', 'Importer'",
            "Usually found in a dedicated section",
            "May be labeled 'Sold To' or 'Customer'"
        ],
        aliases=["Buyer", "Consignee", "Bill To", "Ship To", "Importer", "Customer"],
        required=True
    ),
    "buyer_address": FieldConfig(
        name="buyer_address",
        display_name="Buyer Address",
        field_type=FieldType.STRING,
        description="Complete address of the buyer",
        extraction_guidelines=[
            "Extract full address including street, city, state, country, postal code",
            "Usually directly below buyer name"
        ],
        aliases=["Buyer Address", "Consignee Address", "Ship To Address"]
    ),
    "invoice_currency": FieldConfig(
        name="invoice_currency",
        display_name="Currency",
        field_type=FieldType.CURRENCY,
        description="Currency code for the invoice",
        extraction_guidelines=[
            "Look for 3-letter ISO currency code (USD, EUR, INR, etc.)",
            "May be shown as symbol ($, €, ₹) - convert to ISO code",
            "Usually near total amount or in a currency field"
        ],
        aliases=["Currency", "Curr", "Ccy", "Invoice Currency"],
        validation_rules=["Must be valid ISO 4217 currency code"]
    ),
    "total_amount": FieldConfig(
        name="total_amount",
        display_name="Total Amount",
        field_type=FieldType.NUMBER,
        description="Grand total invoice amount",
        extraction_guidelines=[
            "Look for 'Total', 'Grand Total', 'Net Total', 'Total Amount'",
            "Extract numeric value only (no currency symbols)",
            "Usually the largest amount at the bottom of invoice",
            "If multiple totals exist, use the final/grand total"
        ],
        aliases=["Total", "Grand Total", "Net Total", "Total Amount", "Invoice Total"],
        required=True,
        validation_rules=["Must be a positive number", "Should match sum of line items if present"]
    ),
    "invoice_toi": FieldConfig(
        name="invoice_toi",
        display_name="Terms of Invoice (IncoTerms)",
        field_type=FieldType.STRING,
        description="International Commercial Terms (IncoTerms)",
        extraction_guidelines=[
            "Look for IncoTerms like: EXW, FCA, FAS, FOB, CFR, CIF, CPT, CIP, DAP, DPU, DDP",
            "Usually labeled 'Terms', 'IncoTerms', 'Delivery Terms', 'Trade Terms'",
            "May include location (e.g., 'FCA Singapore', 'CIP Mumbai')",
            "Extract only the IncoTerm code",
            "If IncoTerm is foud return Incoterm If no inco term is found return null"
        ],
        aliases=["Terms", "IncoTerms", "Delivery Terms", "Trade Terms", "Terms of Delivery"],
        validation_rules=["Must be valid IncoTerm: EXW, FCA, FAS, FOB, CFR, CIF, CPT, CIP, DAP, DPU, DDP"]
    ),
    "invoice_exchange_rate": FieldConfig(
        name="invoice_exchange_rate",
        display_name="Exchange Rate",
        field_type=FieldType.NUMBER,
        description="Currency exchange rate if mentioned",
        extraction_guidelines=[
            "Look for 'Exchange Rate', 'FX Rate', 'Conversion Rate'",
            "Extract numeric value only"
        ],
        aliases=["Exchange Rate", "FX Rate", "Conversion Rate", "Rate of Exchange"]
    ),
    "invoice_po_date": FieldConfig(
        name="invoice_po_date",
        display_name="PO Date",
        field_type=FieldType.DATE,
        description="Purchase Order date if referenced",
        extraction_guidelines=[
            "Look for 'PO Date', 'Purchase Order Date', 'Order Date'",
            "May be found in reference section",
            "Return in ISO format: YYYY-MM-DD"
        ],
        aliases=["PO Date", "Purchase Order Date", "Order Date"]
    ),
}

# =============================================================================
# INVOICE LINE ITEM FIELD CONFIGURATION
# =============================================================================
INVOICE_ITEM_FIELDS: Dict[str, FieldConfig] = {
    "item_no": FieldConfig(
        name="item_no",
        display_name="Item Number",
        field_type=FieldType.STRING,
        description="Serial/line number of the item",
        extraction_guidelines=[
            "HARDCODED RULE: ALWAYS return the value '1' for this field.",
            "Do not extract from the document.",
            "Ignore any other serial numbers visually present."
        ],
        aliases=["S.No", "Sr.No", "Item No", "Line No", "#", "Sl No"],
        default="1"
    ),
    "item_description": FieldConfig(
        name="item_description",
        display_name="Description",
        field_type=FieldType.STRING,
        description="Description of the goods/product",
        extraction_guidelines=[
            "Look for 'Description', 'Particulars', 'Item Description', 'Goods'",
            "Extract complete description including any specifications",
            "May span multiple lines - combine into single string",
            "**CRITICAL ROW ALIGNMENT RULE:** Use visual alignment to ensure the Description belongs to the correct Part Number row.",
            "Do not 'carry over' text from previous rows.",
            "If a description is missing for a line item, strictly return null."
        ],
        aliases=["Description", "Particulars", "Item Description", "Goods", "Product"],
        required=True
    ),
    "item_part_no": FieldConfig(
        name="item_part_no",
        display_name="Part Number",
        field_type=FieldType.STRING,
        description="Part number/SKU of the item",
        extraction_guidelines=[
            "Look for 'Part No', 'Part Number', 'P/N', 'SKU', 'Item Code'",
            "Usually alphanumeric code",
            "May be embedded within description"
        ],
        aliases=["Part No", "Part Number", "P/N", "SKU", "Item Code", "Material No"]
    ),
    # --- UPDATED FIELD: STRICT KEYWORD MATCHING ONLY ---
    "item_po_no": FieldConfig(
        name="item_po_no",
        display_name="PO Number",
        field_type=FieldType.STRING,
        description="Purchase Order number for this line item",
        extraction_guidelines=[
            "**CRITICAL: STRICT PO MATCHING**",
            "ONLY extract if you find these EXACT keywords:",
            "  1. 'PO No' or 'PO No.' or 'PO No:'",
            "  2. 'PO Number' or 'PO Number:'",
            "  3. 'Purchase Order' or 'Purchase Order No'",
            "",
            "**EXCLUSION RULES:**",
            "  - IGNORE 'Order No' or 'Order Number' (unless preceded by 'Purchase' or 'PO')",
            "  - IGNORE 'Buyers Order No'",
            "  - IGNORE 'Sales Order No'",
            "",
            "If no explicit 'PO' or 'Purchase Order' label exists, return NULL."
        ],
        aliases=["PO No", "PO Number", "Purchase Order"] # Removed "Order No"
    ),
    # ---------------------------------------------------
    "item_date": FieldConfig(
        name="item_date",
        display_name="Item Date",
        field_type=FieldType.DATE,
        description="Date specific to the line item (e.g. Delivery Date)",
        extraction_guidelines=[
            "Look for columns like 'Date', 'Delivery Date', 'Service Date', 'Shipment Date' in the item table",
            "Return in ISO format: YYYY-MM-DD",
            "If no specific date is listed for the item, return null"
        ],
        aliases=["Date", "Delivery Date", "Service Date", "Ship Date"],
        required=False
    ),
    "hsn_code": FieldConfig(
        name="hsn_code",
        display_name="HSN/HS Code",
        field_type=FieldType.STRING,
        description="Harmonized System Code (HS Code) for customs",
        extraction_guidelines=[
            "Look for 'HSN', 'HS Code', 'Tariff Code'",
            "Usually 4-10 digit code",
            "Critical for customs - ensure accuracy",
            "May be same as CTH or RITC in some documents"
        ],
        aliases=["HSN", "HS Code", "Tariff Code", "HSN Code", "Commodity Code"],
        validation_rules=["Must be 4-10 digits", "Should be valid HS Code format"]
    ),
    "item_cth": FieldConfig(
        name="item_cth",
        display_name="CTH (Customs Tariff Heading)",
        field_type=FieldType.STRING,
        description="Customs Tariff Heading code",
        extraction_guidelines=[
            "Look for 'CTH', 'Customs Tariff Heading', 'Tariff Heading'",
            "Usually 8 digit code",
            "May be same as or related to HSN code"
        ],
        aliases=["CTH", "Customs Tariff Heading", "Tariff Heading"]
    ),
    "item_ritc": FieldConfig(
        name="item_ritc",
        display_name="RITC Code",
        field_type=FieldType.STRING,
        description="Regional Import Tariff Code",
        extraction_guidelines=[
            "Look for 'RITC', 'RITC Code'",
            "Usually 8 digit code similar to CTH",
            "Used for Indian customs classification"
        ],
        aliases=["RITC", "RITC Code"]
    ),
    "item_ceth": FieldConfig(
        name="item_ceth",
        display_name="CETH Code",
        field_type=FieldType.STRING,
        description="Central Excise Tariff Heading code",
        extraction_guidelines=[
            "Look for 'CETH', 'Central Excise Tariff'",
            "Usually 8 digit code",
            "May be same as CTH in many cases"
        ],
        aliases=["CETH", "Central Excise Tariff", "CETH Code"]
    ),
    "item_quantity": FieldConfig(
        name="item_quantity",
        display_name="Quantity",
        field_type=FieldType.NUMBER,
        description="Number of units",
        extraction_guidelines=[
            "Look for 'Qty', 'Quantity', 'Nos', 'Units'",
            "Extract numeric value only"
        ],
        aliases=["Qty", "Quantity", "Nos", "Units", "Pcs"],
        required=True
    ),
    "item_uom": FieldConfig(
        name="item_uom",
        display_name="Unit of Measure",
        field_type=FieldType.STRING,
        description="Unit of measurement",
        extraction_guidelines=[
            "Look for 'UOM', 'Unit', 'Unit of Measure'",
            "Common values: PCS, NOS, KG, MTR, SET, EA, PC"
        ],
        aliases=["UOM", "Unit", "Unit of Measure"]
    ),
    "item_unit_price": FieldConfig(
        name="item_unit_price",
        display_name="Unit Price",
        field_type=FieldType.NUMBER,
        description="Price per unit",
        extraction_guidelines=[
            "Look for 'Rate', 'Unit Price', 'Price', 'Unit Rate'",
            "Extract numeric value only (no currency)",
            "Handle European format (comma as decimal separator)"
        ],
        aliases=["Rate", "Unit Price", "Price", "Unit Rate", "Price/Unit"]
    ),
    "item_amount": FieldConfig(
        name="item_amount",
        display_name="Line Amount",
        field_type=FieldType.NUMBER,
        description="Total amount for this line item (qty × rate)",
        extraction_guidelines=[
            "Look for 'Amount', 'Total', 'Line Total', 'Value'",
            "Usually last column in items table",
            "Should equal quantity × unit price"
        ],
        aliases=["Amount", "Total", "Line Total", "Value", "Net Amount"]
    ),
    "item_origin_country": FieldConfig(
        name="item_origin_country",
        display_name="Country of Origin",
        field_type=FieldType.STRING,
        description="Country where goods were manufactured",
        extraction_guidelines=[
            "Look for 'COO', 'Country of Origin', 'Origin', 'Made In'",
            "May be full country name or ISO code (US, DE, CN, TW, etc.)",
            "May be include 'Made in' prefix - extract country name only",
            "May be same for all items"
        ],
        aliases=["COO", "Country of Origin", "Origin", "Made In", "Manufacturing Country"],
        validation_rules=["Should be valid country name or ISO code"]
    ),
    "item_mfg_name": FieldConfig(
        name="item_mfg_name",
        display_name="Manufacturer Name",
        field_type=FieldType.STRING,
        description="Name of the manufacturer company",
        extraction_guidelines=[
            "Look for 'Manufacturer', 'Mfg', 'Made By', 'Producer'",
            "May be different from seller/supplier",
            "Extract full company name"
        ],
        aliases=["Manufacturer", "Mfg", "Mfg Name", "Made By", "Producer", "Manufacturing Company"]
    ),
    "item_mfg_addr": FieldConfig(
        name="item_mfg_addr",
        display_name="Manufacturer Address",
        field_type=FieldType.STRING,
        description="Complete address of the manufacturer",
        extraction_guidelines=[
            "Look for manufacturer address near manufacturer name",
            "Extract full address including street, city, country",
            "May be same as seller address in some cases"
        ],
        aliases=["Manufacturer Address", "Mfg Address", "Factory Address"]
    ),
    "item_mfg_country": FieldConfig(
        name="item_mfg_country",
        display_name="Manufacturer Country",
        field_type=FieldType.STRING,
        description="Country where manufacturer is located",
        extraction_guidelines=[
            "Extract country from manufacturer address",
            "May be different from Country of Origin",
            "Use full country name or ISO code"
        ],
        aliases=["Manufacturer Country", "Mfg Country", "Factory Country"]
    ),
}

# =============================================================================
# AIRWAY BILL FIELD CONFIGURATION
# Note: Field names match production data format (frieght spelling is intentional)
# =============================================================================
AIRWAY_BILL_FIELDS: Dict[str, FieldConfig] = {
    "master_awb_no": FieldConfig(
        name="master_awb_no",
        display_name="Master AWB Number",
        field_type=FieldType.STRING,
        description="Master Airway Bill number",
        extraction_guidelines=[
            "Look for 'MAWB', 'Master AWB', 'Master Airway Bill'",
            "Usually 11 digits in format: XXX-XXXXXXXX",
            "First 3 digits are airline code",
            "Extract digits only, remove hyphens for output"
        ],
        aliases=["MAWB", "Master AWB", "Master Airway Bill", "Master AWB No"],
        required=True,
        validation_rules=["Should be numeric, typically 11 digits"]
    ),
    "house_awb_no": FieldConfig(
        name="house_awb_no",
        display_name="House AWB Number",
        field_type=FieldType.STRING,
        description="House Airway Bill number",
        extraction_guidelines=[
            "Look for 'HAWB', 'House AWB', 'House Airway Bill'",
            "Assigned by freight forwarder",
            "Format varies by forwarder (e.g., SIN10076768, NEB06511186)"
        ],
        aliases=["HAWB", "House AWB", "House Airway Bill", "House AWB No"]
    ),
    "shipper_name": FieldConfig(
        name="shipper_name",
        display_name="Shipper Name",
        field_type=FieldType.STRING,
        description="Name of the shipper/sender",
        extraction_guidelines=[
            "Look for 'Shipper', 'Sender', 'Consignor', 'Exporter'",
            "Usually in the shipper details box"
        ],
        aliases=["Shipper", "Sender", "Consignor", "Exporter"]
    ),
    "pkg_in_qty": FieldConfig(
        name="pkg_in_qty",
        display_name="Package Quantity",
        field_type=FieldType.NUMBER,
        description="Number of packages/pieces",
        extraction_guidelines=[
            "Look for 'No. of Pieces', 'Packages', 'Pcs', 'Pkg', 'Pieces'",
            "Extract numeric value only"
        ],
        aliases=["No. of Pieces", "Packages", "Pcs", "Pkg", "Number of Packages", "Pieces"],
        required=True
    ),
    "gross_weight_kg": FieldConfig(
        name="gross_weight_kg",
        display_name="Gross Weight (KG)",
        field_type=FieldType.NUMBER,
        description="Total gross weight in kilograms",
        extraction_guidelines=[
            "Look for 'Gross Weight', 'Weight', 'Actual Weight', 'Gross Wt'",
            "Extract numeric value",
            "Include decimal if present"
        ],
        aliases=["Gross Weight", "Weight", "Actual Weight", "Gross Wt", "G.W."],
        required=True
    ),
    "chargeable_weight_kg": FieldConfig(
        name="chargeable_weight_kg",
        display_name="Chargeable Weight (KG)",
        field_type=FieldType.NUMBER,
        description="Chargeable weight in kilograms",
        extraction_guidelines=[
            "Look for 'Chargeable Weight', 'Chrg Weight', 'Billable Weight'",
            "May be higher than gross weight due to dimensional weight",
            "Extract numeric value"
        ],
        aliases=["Chargeable Weight", "Chrg Weight", "Billable Weight", "Chrg Wt", "Ch. Wt."],
        required=True
    ),
    "total_frieght": FieldConfig(
        name="total_frieght",
        display_name="Total Freight",
        field_type=FieldType.NUMBER,
        description="Total freight charges",
        extraction_guidelines=[
            "Look for 'Total Freight', 'Freight Charges', 'Air Freight', 'Total Prepaid'",
            "Extract numeric value only (no currency symbols)"
        ],
        aliases=["Total Freight", "Freight Charges", "Air Freight", "Freight", "Total Prepaid", "Total Collect"]
    ),
    "frieght_currency": FieldConfig(
        name="frieght_currency",
        display_name="Freight Currency",
        field_type=FieldType.CURRENCY,
        description="Currency for freight charges",
        extraction_guidelines=[
            "Look for currency code near freight amount",
            "Use 3-letter ISO code (USD, EUR, SGD, etc.)",
            "May be in a separate currency field or next to freight amount"
        ],
        aliases=["Currency", "Curr", "Ccy", "Freight Currency"]
    ),
}


# =============================================================================
# PROMPT GENERATOR
# =============================================================================
def generate_field_prompt_section(fields: Dict[str, FieldConfig], section_name: str = "FIELDS") -> str:
    """
    Generate a structured prompt section from field configurations.
    """
    lines = [f"### {section_name} TO EXTRACT:"]
    
    for field_name, config in fields.items():
        required_tag = " [REQUIRED]" if config.required else ""
        lines.append(f"\n- **{config.name}** ({config.display_name}){required_tag}")
        lines.append(f"  - Description: {config.description}")
        
        if config.aliases:
            lines.append(f"  - Look for: {', '.join([f'\"{a}\"' for a in config.aliases[:5]])}")
        
        if config.extraction_guidelines:
            lines.append("  - Guidelines:")
            for guideline in config.extraction_guidelines:
                lines.append(f"    • {guideline}")
        
        if config.validation_rules:
            lines.append("  - Validation:")
            for rule in config.validation_rules:
                lines.append(f"    • {rule}")
    
    return "\n".join(lines)


def generate_extraction_prompt(
    doc_type: str,
    fields: Dict[str, FieldConfig],
    item_fields: Optional[Dict[str, FieldConfig]] = None,
    custom_instructions: Optional[List[str]] = None
) -> str:
    """
    Generate a complete extraction prompt from field configurations.
    """
    prompt_parts = []
    
    # Header
    if doc_type == "invoice":
        prompt_parts.append("You are a forensic Customs Data Extractor. Extract data for Customs Filing from this Invoice.")
    else:
        prompt_parts.append("You are a forensic Customs Data Extractor. Extract data from this Airway Bill.")
    
    prompt_parts.append("\n")
    
    # Main fields
    prompt_parts.append(generate_field_prompt_section(fields, "DOCUMENT FIELDS"))
    
    # Item fields (for invoice)
    if item_fields:
        prompt_parts.append("\n")
        prompt_parts.append(generate_field_prompt_section(item_fields, "LINE ITEM FIELDS"))
        prompt_parts.append("\n### LINE ITEMS INSTRUCTIONS:")
        prompt_parts.append("- Extract ALL line items from the items/products table")
        prompt_parts.append("- Each row in the table = one item in the 'items' array")
        prompt_parts.append("- If manufacturer details are the same for all items, still include them in each item")
    
    # Custom instructions
    if custom_instructions:
        prompt_parts.append("\n### ADDITIONAL INSTRUCTIONS:")
        for instruction in custom_instructions:
            prompt_parts.append(f"- {instruction}")
    
    # Output instruction
    prompt_parts.append("\n### OUTPUT:")
    prompt_parts.append("Return valid JSON strictly matching the provided schema.")
    prompt_parts.append("Use null for fields not found in the document.")
    
    return "\n".join(prompt_parts)


# =============================================================================
# PRE-BUILT PROMPTS (Enhanced with Business Logic)
# =============================================================================

# Custom high-accuracy invoice prompt with all business rules
INVOICE_PROMPT = """
You are a forensic Customs Data Extractor. Extract data for Customs Filing from this Commercial Invoice.

### CRITICAL BUSINESS RULES:

#### 1. EUROPEAN DECIMAL FORMAT (VERY IMPORTANT)
Many European invoices use comma as decimal separator and period as thousands separator:
- `74,932` means **74.932** (seventy-four point nine three two)
- `1.685,84` means **1685.84** (one thousand six hundred eighty-five point eighty-four)
- `10.000,00` means **10000.00** (ten thousand)
**ALWAYS convert to standard format**: Use dot (.) for decimals, NO thousands separators.

#### 2. INCOTERMS EXTRACTION
Look for terms: EXW, FCA, FAS, FOB, CFR, CIF, CPT, CIP, DAP, DPU, DDP
- Usually labeled "Terms", "IncoTerms", "Delivery Terms"
- May include location (e.g., "FCA Singapore", "CIP Mumbai")
- Extract ONLY the IncoTerm code (e.g., "FCA", "CIP", "FOB")

#### 3. PRODUCT DESCRIPTION (CRITICAL)
**item_description = [Description Text] + [Part Number/Article Number]**
- Concatenate the description with the part number
- Example: If description is "Ceramic Capacitor" and part no is "114300", 
  item_description should be "Ceramic Capacitor 114300"
- **CRITICAL ROW ALIGNMENT RULE:** Use visual alignment to ensure the Description belongs to the correct Part Number row. Do not 'carry over' text from previous rows. If a description is missing, return null.

#### 4. HS CODE / CTH / RITC / CETH (VERY IMPORTANT)
These tariff codes are CRITICAL for customs:
- Look for columns: "HSN", "HS Code", "CTH", "RITC", "CETH", "Tariff Code", "Customs Tariff"
- **hsn_code**: The main HS/Tariff code (e.g., "84671110", "39173300")
- **item_cth**: Customs Tariff Heading - usually SAME as hsn_code
- **item_ritc**: Regional Import Tariff Code - usually SAME as hsn_code  
- **item_ceth**: Central Excise Tariff Heading - usually SAME as hsn_code
- **If only one tariff code column exists, copy that value to ALL four fields (hsn_code, item_cth, item_ritc, item_ceth)**
- These codes are 6-8 digit numbers (e.g., "84671110", "39173300")

#### 5. MANUFACTURER INFORMATION (IMPORTANT)
For commercial invoices from distributors/wholesalers:
- **item_mfg_name**: Look for "Manufacturer", "Produced by", "Made by"
- **If the seller/shipper is ALSO the manufacturer** (common for parts distributors like Atlas Copco, Power Tools Distribution): 
  - Use the SELLER NAME as item_mfg_name
  - Use the SELLER ADDRESS as item_mfg_addr
  - Extract country from seller address as item_mfg_country
- Example: If seller is "Power Tools Distribution N.V., Industrielaan 40, Belgium":
  - item_mfg_name = "Power Tools Distribution N.V."
  - item_mfg_addr = "Industrielaan 40, 3730 Bilzen-Hoeselt, Belgium"
  - item_mfg_country = "Belgium"

#### 6. QUANTITY & UNIT PRICE
- Look for "Qty", "Quantity", "Delivered qty"
- Unit Price may be labeled "Rate", "Unit Price", "Price/Unit"
- **Apply European decimal conversion if needed**
- If only Total Amount is given: Unit Price = Total / Quantity

#### 7. COUNTRY OF ORIGIN
- Look for "COO", "Country of Origin", "Origin", "Made In"
- May be 2-letter code (TW, DE, JP) or full name (Taiwan, Germany, Japan)
- Extract as shown in document

#### 8. PURCHASE ORDER
- Look for "PO No", "P.O.", "Purchase Order", "Order No"
- **invoice_po_date**: Date of the PO (if shown)
- **item_po_no**: PO number per line item (may be same for all items)

### DOCUMENT FIELDS TO EXTRACT:
- **invoice_number**: Invoice No, Invoice #, Bill No
- **invoice_date**: Return in YYYY-MM-DD format
- **seller_name**: Supplier/Vendor/Exporter name (company in letterhead)
- **seller_address**: Full address of seller
- **buyer_name**: Buyer/Consignee/Bill To/Importer
- **buyer_address**: Full address of buyer
- **invoice_currency**: 3-letter code (USD, EUR, JPY, SGD, etc.)
- **total_amount**: Grand total (apply decimal conversion)
- **invoice_toi**: IncoTerms (FCA, FOB, CIF, etc.)
- **invoice_exchange_rate**: If mentioned on invoice
- **invoice_po_date**: PO Date if referenced (YYYY-MM-DD format)

### LINE ITEM FIELDS (Extract for EACH item row):
- **item_no**: HARDCODED RULE - ALWAYS set to "1"
- **item_description**: Description + Part Number concatenated (Correctly aligned to row)
- **item_part_no**: Part Number/Article Number/SKU
- **item_po_no**: Purchase Order number for this item
- **item_date**: Specific date for the item (YYYY-MM-DD)
- **hsn_code**: HS Code / Tariff Code (copy to CTH/RITC/CETH if only one code exists)
- **item_cth**: Customs Tariff Heading (same as hsn_code if not separate)
- **item_ritc**: Regional Import Tariff Code (same as hsn_code if not separate)
- **item_ceth**: Central Excise Tariff Heading (same as hsn_code if not separate)
- **item_quantity**: Quantity (numeric, apply decimal conversion)
- **item_uom**: Unit of Measure (PC, PCS, NOS, KG, etc.)
- **item_unit_price**: Price per unit (numeric, apply decimal conversion)
- **item_amount**: Line total (numeric, apply decimal conversion)
- **item_origin_country**: Country of Origin (2-letter code or full name)
- **item_mfg_name**: Manufacturer name (use seller if they are the manufacturer/distributor)
- **item_mfg_addr**: Manufacturer address (use seller address if applicable)
- **item_mfg_country**: Manufacturer country

### OUTPUT RULES:
1. Return valid JSON matching the schema exactly
2. Use **null** ONLY for fields truly not present in the document
3. Extract ALL line items from the products/items table
4. Apply European decimal conversion to ALL numeric values
5. Dates must be in YYYY-MM-DD format
6. For tariff codes: If only hsn_code exists, copy it to item_cth, item_ritc, item_ceth
"""

# Custom high-accuracy AWB prompt with business rules and specific locations
AWB_PROMPT = """
You are a forensic Customs Data Extractor. Extract data from this Air Waybill (AWB) document.

### DOCUMENT TYPES:
1. **Master AWB (MAWB)**: Issued by airline, 11 digits exactly
2. **House AWB (HAWB)**: Issued by freight forwarder, alphanumeric format

### FIELD LOCATIONS (CRITICAL):

#### 1. MASTER AIRWAY BILL NUMBER
- **Location**: TOP LEFT CORNER of the document
- **Format**: 11 DIGITS ONLY (e.g., "17617629931")
- May also appear in CENTER under "Accounting Information"
- May be labeled "M Airway Bill", "Master AWB", "MAWB No"
- **IMPORTANT**: If the number contains text prefixes like "SIN", "NEU", "NEB" - IGNORE THE TEXT, extract ONLY the numeric digits
- Example: "SIN17617629931" → extract as "17617629931"
- Example: "176-17629931" → extract as "17617629931"
- Return DIGITS ONLY (no letters, hyphens, or spaces)

#### 2. HOUSE AIRWAY BILL NUMBER  
- **Location**: TOP RIGHT CORNER of the document
- **Format**: Alphanumeric, typically 8+ characters (e.g., "SIN10076768", "NEB06511186")
- May be labeled "House Airway Bill", "HAWB", "H AWB No"
- Keep alphanumeric characters

#### 3. PACKAGE QUANTITY
- Look for "No. of Pieces", "Packages", "Pcs", "Pieces"
- Unit is typically "pcs" (pieces)
- Extract as integer number

#### 4. GROSS WEIGHT (KG)
- **Location**: Weight section/box
- Look for "Gross Weight", "G.W.", "Actual Weight", "Gross Wt (Kgs)"
- Extract numeric value ONLY (without "kg" or "kgs")

#### 5. CHARGEABLE WEIGHT (KG)
- **Location**: Weight section, may be near "Warehouse" label
- Look for "Chargeable Weight", "Chrg Wt", "Ch. Weight", "Billable Weight"
- May be HIGHER than gross weight (due to volumetric calculation)
- Extract numeric value ONLY

#### 6. TOTAL FREIGHT
- **Location**: Charges section
- Look for "Total Freight", "Total Collect", "Total Prepaid", "Air Freight"
- May be labeled simply as freight amount
- Extract numeric value ONLY (no currency symbol)

#### 7. FREIGHT CURRENCY
- **Location**: Near freight amount
- **Format**: 3-letter ISO code (SGD, EUR, USD, JPY, etc.)
- May be in separate "Currency" field or printed next to amount

### FIELDS TO EXTRACT:
- **master_awb_no**: 11-digit Master AWB (TOP LEFT corner)
- **house_awb_no**: House AWB alphanumeric (TOP RIGHT corner)
- **shipper_name**: Shipper/Sender company name
- **pkg_in_qty**: Package quantity (integer)
- **gross_weight_kg**: Gross weight in KG (decimal)
- **chargeable_weight_kg**: Chargeable weight in KG (decimal)
- **total_frieght**: Total freight amount (decimal, "Total collect" or "Total prepaid")
- **frieght_currency**: 3-letter currency code (SGD, EUR, USD, etc.)

### OUTPUT RULES:
1. Return valid JSON matching the schema exactly
2. Use **null** for any field not found in the document
3. All weights and freight should be numeric (float)
4. MAWB must be 11 digits only (remove hyphens)
5. Currency must be 3-letter ISO code
"""

CLASSIFICATION_PROMPT = """
You are a Senior Document Router. Analyze this PDF.

### MISSION:
Identify page ranges for each document type:

1. **INVOICE**
   - Look for: "Invoice No", "Tax Invoice", "Commercial Invoice", "Total Amount", "Bill To"
   - Contains: item tables, prices, quantities, totals

2. **AIRWAY BILL**
   - Look for: "Master AWB", "House AWB", "MAWB", "HAWB", "Shipper", "Consignee"
   - Contains: shipping details, weights, package counts

### RULES:
- IGNORE these document types: "Shipping Instructions", "Packing List", "Specification Sheet", "Certificate"
- Return strictly valid page numbers (1-indexed)
- Multi-page documents: Return the [start, end] range
- Single page documents: Return [page, page]

### OUTPUT JSON:
{
  "invoices": [[start, end], ...],
  "airway_bills": [[start, end], ...]
}
"""


# =============================================================================
# UTILITY: Get field as JSON config (for debugging/export)
# =============================================================================
def export_fields_as_json(fields: Dict[str, FieldConfig]) -> dict:
    """Export field configuration as JSON-serializable dict."""
    return {
        name: {
            "name": cfg.name,
            "display_name": cfg.display_name,
            "type": cfg.field_type.value,
            "description": cfg.description,
            "aliases": cfg.aliases,
            "required": cfg.required,
            "guidelines": cfg.extraction_guidelines,
            "validation": cfg.validation_rules
        }
        for name, cfg in fields.items()
    }


def get_all_field_names() -> dict:
    """Get all field names for reference."""
    return {
        "invoice_fields": list(INVOICE_FIELDS.keys()),
        "item_fields": list(INVOICE_ITEM_FIELDS.keys()),
        "awb_fields": list(AIRWAY_BILL_FIELDS.keys())
    }


if __name__ == "__main__":
    # Test: Print generated prompts
    print("=" * 80)
    print("INVOICE PROMPT:")
    print("=" * 80)
    print(INVOICE_PROMPT)
    print("\n" + "=" * 80)
    print("AWB PROMPT:")
    print("=" * 80)
    print(AWB_PROMPT)
    print("\n" + "=" * 80)
    print("ALL FIELD NAMES:")
    print("=" * 80)
    print(get_all_field_names())