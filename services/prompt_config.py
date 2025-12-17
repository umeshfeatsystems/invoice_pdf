"""
Standardized JSON-based Prompt Configuration System
====================================================
This module defines extraction fields and their guidelines in a structured JSON format.
Updated for Specific Vendor Routing (ABB/Epiroc & Crown) while maintaining Global logic.
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
    """Configuration for a single extraction field."""
    name: str                           # JSON key name
    display_name: str                   # Human readable name
    field_type: FieldType               # Data type
    description: str                    # What this field represents
    extraction_guidelines: List[str]    # How to extract this field
    aliases: List[str] = None           # Alternative names
    required: bool = False              # Is this field mandatory
    default: Any = None                 # Default value
    validation_rules: List[str] = None  # Validation instructions
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.validation_rules is None:
            self.validation_rules = []


# =============================================================================
# INVOICE FIELD CONFIGURATION (GLOBAL)
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
# INVOICE LINE ITEM FIELD CONFIGURATION (GLOBAL)
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
    "item_po_no": FieldConfig(
        name="item_po_no",
        display_name="Item Purchase Order Number",
        field_type=FieldType.STRING,
        description="Customer-facing purchase order reference associated with the item",
        extraction_guidelines=[
            "**PRIMARY RULE (HIGHEST PRIORITY)**:",
            "If a value is present next to any customer-facing labels such as:",
            "- 'Your reference'",
            "- 'Your order no', 'Your order number'",
            "- 'Customer order', 'Customer Order No'",
            "- 'Customer PO', 'Customer P.O.'",
            "then extract THIS value as item_po_no.",
            "This rule overrides proximity, repetition, and numeric confidence.",
            "",
            "**SECONDARY RULE (ONLY IF NO TIER 1 VALUE EXISTS)**:",
            "If no customer-facing reference exists, extract values labeled as:",
            "- 'Order no', 'Order number', or 'Sales order'.",
            "",
            "**ANTI-HALLUCINATION RULES**:",
            "Do NOT prefer numeric-only values over alphanumeric ones.",
            "If both Tier 1 and Tier 2 values exist, ALWAYS choose Tier 1.",
            "",
            "**NULL RULE**:",
            "If no valid order reference exists, return null."
        ],
        aliases=["Your reference", "Customer PO", "Order No"]
    ),
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
# SPECIFIC VENDOR OVERRIDES
# =============================================================================

# 1. ABB / Epiroc Logic
ABB_ITEM_PO_FIELD = FieldConfig(
    name="item_po_no",
    display_name="Item PO Number (ABB/Epiroc)",
    field_type=FieldType.STRING,
    description="Customer Reference / Order Number",
    extraction_guidelines=[
        "**STRICT TARGETING RULE**:",
        "Extract values ONLY from these labels:",
        "- 'Your reference'",
        "- 'Your orderNo.'",
        "- 'Our order no'",
        "If these specific labels are not found, return null.",
        "**ANTI-HALLUCINATION**: Do NOT guess based on proximity."
    ],
    aliases=["Your reference", "Your orderNo.", "Our order no"]
)

# 2. Crown Logic
CROWN_ITEM_PO_FIELD = FieldConfig(
    name="item_po_no",
    display_name="Item PO Number (Crown)",
    field_type=FieldType.STRING,
    description="Customer Purchase Order",
    extraction_guidelines=[
        "**STRICT TARGETING RULE**:",
        "Extract value ONLY from label: 'Customer P.O.'",
        "**NEGATIVE RULE (IGNORE)**:",
        "Do NOT extract 'Order Number'",
        "Do NOT extract 'Shipping number'",
        "Do NOT extract 'Packing Slip'",
        "If 'Customer P.O.' is missing, return null."
    ],
    aliases=["Customer P.O."]
)

def create_vendor_fields(override_po_config):
    """Deep copy global fields and inject specific PO override."""
    fields = copy.deepcopy(INVOICE_ITEM_FIELDS)
    fields["item_po_no"] = override_po_config
    return fields

ABB_ITEM_FIELDS = create_vendor_fields(ABB_ITEM_PO_FIELD)
CROWN_ITEM_FIELDS = create_vendor_fields(CROWN_ITEM_PO_FIELD)


# =============================================================================
# PROMPT GENERATION
# =============================================================================

def generate_field_prompt_section(fields: Dict[str, FieldConfig], section_name: str = "FIELDS") -> str:
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
    prompt_parts = []
    
    if "Airway Bill" in doc_type:
        prompt_parts.append(f"You are a forensic Customs Data Extractor. Extract data from this {doc_type}.")
    else:
        prompt_parts.append(f"You are a forensic Customs Data Extractor. Extract data for Customs Filing from this {doc_type}.")
    
    prompt_parts.append("\n")
    prompt_parts.append(generate_field_prompt_section(fields, "DOCUMENT FIELDS"))
    
    if item_fields:
        prompt_parts.append("\n")
        prompt_parts.append(generate_field_prompt_section(item_fields, "LINE ITEM FIELDS"))
        prompt_parts.append("\n### LINE ITEMS INSTRUCTIONS:")
        prompt_parts.append("- Extract ALL line items from the items/products table")
        prompt_parts.append("- **item_no**: HARDCODED RULE: ALWAYS set to '1'")
        prompt_parts.append("- Each row in the table = one item in the 'items' array")
    
    if custom_instructions:
        prompt_parts.append("\n### CRITICAL BUSINESS RULES & INSTRUCTIONS:")
        for instruction in custom_instructions:
            prompt_parts.append(f"- {instruction}")
            
    prompt_parts.append("\n### OUTPUT:")
    prompt_parts.append("Return valid JSON strictly matching the provided schema.")
    prompt_parts.append("Use null for fields not found in the document.")
    
    return "\n".join(prompt_parts)


# =============================================================================
# FINAL PROMPTS
# =============================================================================

# 1. CLASSIFICATION PROMPT (New Categories)
CLASSIFICATION_PROMPT = """
You are a Senior Document Router. Analyze this PDF and classify page ranges by document type.

### DOCUMENT CATEGORIES:

1. **ABB / Epiroc INVOICE** ("abb_invoices")
   - Look for logo/text: "ABB", "Epiroc"
   - Look for specific labels: "Your reference", "Your orderNo.", "Our order no"

2. **CROWN INVOICE** ("crown_invoices")
   - Look for logo/text: "Crown Worldwide", "Crown"
   - Look for specific labels: "Customer P.O."

3. **STANDARD INVOICE** ("invoices")
   - Any other Commercial Invoice not matching above.
   - Look for: "Invoice No", "Tax Invoice", "Bill To"

4. **AIRWAY BILL** ("airway_bills")
   - Look for: "Master AWB", "MAWB", "HAWB", "Shipper", "Consignee"

### RULES:
- IGNORE: "Packing List", "Instructions", "Certificates"
- Return strictly valid page numbers (1-indexed).
- Multi-page docs: Return [start, end].
- Single page: Return [page, page].

### OUTPUT JSON:
{
  "invoices": [[start, end], ...],
  "abb_invoices": [[start, end], ...],
  "crown_invoices": [[start, end], ...],
  "airway_bills": [[start, end], ...]
}
"""

# 2. GLOBAL INVOICE PROMPT (Standard Logic)
INVOICE_PROMPT = generate_extraction_prompt(
    "Commercial Invoice", INVOICE_FIELDS, INVOICE_ITEM_FIELDS,
    custom_instructions=[
        "EUROPEAN FORMAT: 1.000,00 = 1000.00 (Convert to standard)",
        "DATES: ISO Format YYYY-MM-DD",
        "DESCRIPTION: Concatenate Description + Part Number",
        "HS CODES: If only one exists, copy to CTH, RITC, CETH",
        "MANUFACTURER: If seller is manufacturer, copy seller details to item_mfg fields"
    ]
)

# 3. ABB / EPIROC PROMPT (Specific Logic)
ABB_PROMPT = generate_extraction_prompt(
    "ABB / Epiroc Invoice", INVOICE_FIELDS, ABB_ITEM_FIELDS,
    custom_instructions=[
        "**VENDOR SPECIFIC**: This is an ABB/Epiroc invoice.",
        "**item_po_no**: STRICTLY extract 'Your reference', 'Your orderNo.', or 'Our order no'.",
        "EUROPEAN FORMAT: Convert 1.000,00 to 1000.00",
        "DATES: ISO Format YYYY-MM-DD"
    ]
)

# 4. CROWN PROMPT (Specific Logic)
CROWN_PROMPT = generate_extraction_prompt(
    "Crown Worldwide Invoice", INVOICE_FIELDS, CROWN_ITEM_FIELDS,
    custom_instructions=[
        "**VENDOR SPECIFIC**: This is a Crown Worldwide invoice.",
        "**item_po_no**: STRICTLY extract 'Customer P.O.'",
        "**IGNORE**: 'Order Number', 'Shipping number', 'Packing Slip' for item_po_no.",
        "DATES: ISO Format YYYY-MM-DD"
    ]
)

# 5. AWB PROMPT
AWB_PROMPT = generate_extraction_prompt(
    "Airway Bill", AIRWAY_BILL_FIELDS, None,
    custom_instructions=[
        "MAWB: 11 Digits only (Top Left)",
        "HAWB: Alphanumeric (Top Right)",
        "Weights/Freight: Numeric only, no units"
    ]
)

if __name__ == "__main__":
    print(INVOICE_PROMPT)