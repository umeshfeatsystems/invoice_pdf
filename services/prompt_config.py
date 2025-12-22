"""
Standardized JSON-based Prompt Configuration System
====================================================
Updated for Specific Vendor Routing + Universal Heuristics for Global Files.

1. Commin (Murata / SIN-...)
2. Type 1 (0100935473...)
3. Type 2 (KM_558...)
4. Crown
5. ABB / Epiroc
6. GLOBAL (Universal Heuristic fallback for random files)
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
        if self.aliases is None:
            self.aliases = []
        if self.validation_rules is None:
            self.validation_rules = []


# =============================================================================
# BASE INVOICE FIELDS (GLOBAL HEURISTICS)
# =============================================================================

INVOICE_FIELDS: Dict[str, FieldConfig] = {
    "invoice_number": FieldConfig(
        name="invoice_number",
        display_name="Invoice Number",
        field_type=FieldType.STRING,
        description="Unique invoice identifier",
        extraction_guidelines=[
            "**ANCHOR LABELS**: 'Invoice No', 'Invoice #', 'Invoice Number', 'Tax Invoice No', 'Bill No', 'Document No'",
            "**LOCATION**: Usually top-right corner of first page, near date",
            "**PRIORITY 1**: Extract value directly after the anchor label",
            "**PRIORITY 2**: If no label, find prominent alphanumeric string near top-right",
            "**FORMAT**: Preserve exact format including prefixes (SIN-, KM_, 0100, etc.)",
            "**NEGATIVE RULE**: Ignore dates, phone numbers, fax numbers, bank account numbers"
        ],
        required=True
    ),

    "invoice_date": FieldConfig(
        name="invoice_date",
        display_name="Invoice Date",
        field_type=FieldType.DATE,
        description="Date the invoice was issued",
        extraction_guidelines=[
            "**ANCHOR LABELS**: 'Invoice Date', 'Date', 'Inv. Date', 'Document Date', 'Issue Date'",
            "**LOCATION**: Usually near invoice number, top section of document",
            "**INPUT FORMATS**: DD/MM/YYYY, MM/DD/YYYY, DD-MMM-YYYY, YYYY-MM-DD",
            "**OUTPUT FORMAT**: Always convert to ISO format YYYY-MM-DD",
            "**PRIORITY**: Prefer date labeled 'Invoice Date' over 'Date' or 'Order Date'",
            "**NEGATIVE RULE**: Ignore 'PO Date', 'Order Date', 'Delivery Date', 'Due Date'"
        ],
        required=True
    ),

    "seller_name": FieldConfig(
        name="seller_name",
        display_name="Seller Name",
        field_type=FieldType.STRING,
        description="Name of the seller/supplier company",
        extraction_guidelines=[
            "**PRIMARY**: Company with LOGO at top of invoice (usually top-left)",
            "**ANCHOR LABELS**: 'From', 'Seller', 'Supplier', 'Vendor', 'Exporter', 'Shipper'",
            "**SECONDARY**: Company name in letterhead or header",
            "**FORMAT**: Extract full legal company name including suffixes (Pte Ltd, Inc, GmbH)",
            "**NEGATIVE RULE**: Not the buyer, not the bank, not the freight forwarder",
            "**NEGATIVE RULE**: Ignore 'Bill To', 'Ship To', 'Consignee' - those are buyers"
        ],
        required=True
    ),

    "seller_address": FieldConfig(
        name="seller_address",
        display_name="Seller Address",
        field_type=FieldType.STRING,
        description="Complete full address of the seller/supplier",
        extraction_guidelines=[
            "**LOCATION**: Directly below or beside Seller Name / Company Logo",
            "**CRITICAL**: Extract the COMPLETE FULL address - do not truncate",
            "**MUST INCLUDE ALL OF THESE (if present)**:",
            "  - Building/Floor number",
            "  - Street name and number",
            "  - Area/District/Suburb",
            "  - City/Town",
            "  - State/Province/Region",
            "  - Postal/ZIP Code",
            "  - Country",
            "**CONCATENATION**: Join multi-line addresses with comma-space (e.g., '123 Main St, Suite 100, Singapore 123456, Singapore')",
            "**OPTIONAL INCLUDE**: Phone, Fax, Email if printed with address",
            "**EXAMPLE OUTPUT**: 'Murata Electronics Singapore Pte Ltd, 1 Harbourfront Avenue, #14-01 Keppel Bay Tower, Singapore 098632'",
            "**NEGATIVE RULE**: Do NOT include bank details, tax IDs, or registration numbers",
            "**NEGATIVE RULE**: Do NOT stop at city - always get postal code and country"
        ]
    ),

    "buyer_name": FieldConfig(
        name="buyer_name",
        display_name="Buyer Name",
        field_type=FieldType.STRING,
        description="Name of the buyer/importer company",
        extraction_guidelines=[
            "**ANCHOR LABELS**: 'Bill To', 'Sold To', 'Buyer', 'Consignee', 'Importer', 'Customer'",
            "**PRIORITY 1**: Value after 'Bill To' or 'Sold To' label",
            "**PRIORITY 2**: If 'Ship To' is different from 'Bill To', use 'Bill To'",
            "**FORMAT**: Extract full legal company name",
            "**NEGATIVE RULE**: NOT the seller, NOT the bank, NOT the freight agent",
            "**NEGATIVE RULE**: Ignore 'Notify Party' - that's different from buyer"
        ],
        required=True
    ),

    "buyer_address": FieldConfig(
        name="buyer_address",
        display_name="Buyer Address",
        field_type=FieldType.STRING,
        description="Complete full address of the buyer/importer",
        extraction_guidelines=[
            "**LOCATION**: Directly below or beside Buyer Name / 'Bill To' section",
            "**CRITICAL**: Extract the COMPLETE FULL address - do not truncate",
            "**MUST INCLUDE ALL OF THESE (if present)**:",
            "  - Building/Floor number",
            "  - Street name and number",
            "  - Area/District/Suburb",
            "  - City/Town",
            "  - State/Province/Region",
            "  - Postal/ZIP Code",
            "  - Country",
            "**CONCATENATION**: Join multi-line addresses with comma-space",
            "**PRIORITY**: Use 'Bill To' address over 'Ship To' address if different",
            "**EXAMPLE OUTPUT**: 'TATA Motors Ltd, Pimpri, Pune 411018, Maharashtra, India'",
            "**NEGATIVE RULE**: Do NOT extract seller address or bank address",
            "**NEGATIVE RULE**: Do NOT stop at city - always get postal code and country"
        ]
    ),

    "invoice_currency": FieldConfig(
        name="invoice_currency",
        display_name="Currency",
        field_type=FieldType.CURRENCY,
        description="3-letter ISO currency code",
        extraction_guidelines=[
            "**ANCHOR LABELS**: 'Currency', 'Curr', 'Amount in', 'All values in'",
            "**LOCATION**: Near Total Amount, or in header section",
            "**FORMAT**: Extract 3-letter ISO code only (USD, EUR, SGD, INR, JPY, GBP, CHF)",
            "**SYMBOL MAPPING**: $ → USD (unless context says SGD/AUD), € → EUR, £ → GBP, ¥ → JPY/CNY",
            "**PRIORITY**: Explicit text 'US Dollars' > Symbol $ alone",
            "**NEGATIVE RULE**: Do NOT guess if no clear indicator exists"
        ]
    ),

    "total_amount": FieldConfig(
        name="total_amount",
        display_name="Total Amount",
        field_type=FieldType.NUMBER,
        description="Grand total invoice value",
        extraction_guidelines=[
            "**ANCHOR LABELS**: 'Total', 'Grand Total', 'Net Total', 'Invoice Total', 'Amount Due', 'Total Amount'",
            "**LOCATION**: Bottom of invoice, after line items",
            "**SELECTOR RULE**: If multiple totals exist, pick the LARGEST value (usually includes tax/freight)",
            "**PRIORITY ORDER**: 'Grand Total' > 'Total' > 'Subtotal'",
            "**FORMAT**: Remove currency symbols ($ € £), remove thousand separators (,), keep decimal point",
            "**NEGATIVE RULE**: Ignore 'Subtotal', 'Items Total' if 'Grand Total' exists",
            "**NEGATIVE RULE**: Ignore 'Balance Due' or 'Amount Paid'"
        ],
        required=True
    ),

    "invoice_toi": FieldConfig(
        name="invoice_toi",
        display_name="Terms of Invoice (IncoTerms)",
        field_type=FieldType.STRING,
        description="International Commercial Terms (Incoterms 2020)",
        extraction_guidelines=[
            "**ANCHOR LABELS**: 'Terms', 'Incoterms', 'Delivery Terms', 'Terms of Delivery', 'Shipping Terms', 'Trade Terms'",
            "**LOCATION**: Usually in header section, near buyer/seller info, or in footer",
            "**VALID VALUES**: EXW, FCA, CPT, CIP, DAP, DPU, DDP, FAS, FOB, CFR, CIF",
            "**FORMAT**: Extract ONLY the 3-letter code (e.g., 'FOB' not 'FOB Shanghai')",
            "**COMMON PATTERNS**: 'FOB Origin', 'CIF Destination', 'EXW Factory'",
            "**PRIORITY**: Look for explicit label first, then scan for 3-letter codes",
            "**NEGATIVE RULE**: Do NOT infer from freight/insurance mentions",
            "**DEFAULT**: Return null if no clear Incoterm found"
        ]
    ),

    "invoice_exchange_rate": FieldConfig(
        name="invoice_exchange_rate",
        display_name="Exchange Rate",
        field_type=FieldType.NUMBER,
        description="Currency exchange rate applied to invoice",
        extraction_guidelines=[
            "**ANCHOR LABELS**: 'Exchange Rate', 'Ex. Rate', 'Conversion Rate', 'Rate', 'FX Rate'",
            "**LOCATION**: Usually near currency or total amount section",
            "**FORMAT**: Numeric value with up to 6 decimal places",
            "**CONTEXT**: Look for pattern like '1 USD = 83.45 INR' → Extract 83.45",
            "**NEGATIVE RULE**: Do NOT confuse with unit price or discount rate",
            "**DEFAULT**: Return null if no exchange rate mentioned (single currency invoice)"
        ]
    ),

    "invoice_po_date": FieldConfig(
        name="invoice_po_date",
        display_name="PO Date",
        field_type=FieldType.DATE,
        description="Purchase Order date referenced in invoice",
        extraction_guidelines=[
            "**ANCHOR LABELS**: 'PO Date', 'Order Date', 'Purchase Order Date', 'Your Order Date'",
            "**LOCATION**: Usually in reference section, near PO number",
            "**FORMAT**: Convert to ISO format YYYY-MM-DD",
            "**PRIORITY**: Date associated with 'PO' or 'Order' reference",
            "**NEGATIVE RULE**: NOT the invoice date, NOT the delivery date, NOT the due date",
            "**DEFAULT**: Return null if no PO date found"
        ]
    ),
}


INVOICE_ITEM_FIELDS: Dict[str, FieldConfig] = {
    "item_no": FieldConfig(
        name="item_no",
        display_name="Item/Serial Number",
        field_type=FieldType.STRING,
        description="Line item sequence number",
        extraction_guidelines=[
            "**COLUMN HEADERS**: 'S.No', 'Sr.', 'Item', 'No.', 'Line', '#', 'Pos'",
            "**LOCATION**: First column of line items table",
            "**FORMAT**: Extract as-is (can be numeric: 1, 2, 3 or alphanumeric: A, B, C)",
            "**DEFAULT**: If no explicit numbering, use row position (1, 2, 3...)",
            "**NEGATIVE RULE**: Not the part number, not the quantity"
        ]
    ),

    "item_description": FieldConfig(
        name="item_description",
        display_name="Description",
        field_type=FieldType.STRING,
        description="Product/service description",
        required=True,
        extraction_guidelines=[
            "**COLUMN HEADERS**: 'Description', 'Particulars', 'Item Description', 'Product', 'Goods', 'Material'",
            "**GRID ALIGNMENT**: Extract text aligned under Description column",
            "**MULTI-LINE RULE**: If a row has description text but NO Quantity/Price, append to PREVIOUS item",
            "**CONCATENATION**: Join wrapped lines with single space",
            "**INCLUDE**: Technical specs, model numbers mentioned in description",
            "**NEGATIVE RULE**: Do NOT create phantom items from text overflow",
            "**NEGATIVE RULE**: Do NOT include standalone page numbers or headers"
        ]
    ),

    "item_part_no": FieldConfig(
        name="item_part_no",
        display_name="Part Number",
        field_type=FieldType.STRING,
        description="Product part number, SKU, or article number",
        extraction_guidelines=[
            "**COLUMN HEADERS**: 'Part No', 'Part Number', 'Article', 'Article No', 'SKU', 'Model', 'Item Code', 'Material No'",
            "**PRIMARY**: Extract from dedicated Part No column",
            "**FALLBACK**: If no column, scan Description for alphanumeric codes like 'AB-1234', '98765-XYZ'",
            "**FORMAT**: Preserve exact format including dashes, spaces, prefixes",
            "**VENDOR SPECIFIC**: For Commin → use 'CUSTOMER PART NUMBER' column",
            "**VENDOR SPECIFIC**: For Type1/Type2 → use 'Article number' column",
            "**NEGATIVE RULE**: Do NOT extract Pack numbers as part numbers",
            "**NEGATIVE RULE**: Do NOT extract model codes (3GAA...) as part numbers for ABB"
        ]
    ),

    "item_po_no": FieldConfig(
        name="item_po_no",
        display_name="Item PO Number",
        field_type=FieldType.STRING,
        description="Purchase Order reference for line item",
        extraction_guidelines=[
            "**COLUMN HEADERS**: 'PO No', 'Order No', 'Your Reference', 'Customer PO', 'Your Order', 'PO Number'",
            "**PRIORITY 1**: Value in dedicated PO column for line item",
            "**PRIORITY 2**: If no line-level PO, use header-level PO for all items",
            "**FORMAT**: Preserve exact format",
            "**VENDOR SPECIFIC**: For Crown → use 'Customer P.O.' field",
            "**VENDOR SPECIFIC**: For ABB → use 'Your orderNo.' or 'Our order no'",
            "**NEGATIVE RULE**: NOT a person's name, NOT a date"
        ]
    ),

    "item_date": FieldConfig(
        name="item_date",
        display_name="Item Date",
        field_type=FieldType.DATE,
        description="Date specific to line item (delivery/ship date)",
        extraction_guidelines=[
            "**COLUMN HEADERS**: 'Date', 'Ship Date', 'Delivery Date', 'Required Date'",
            "**LOCATION**: In line items table, date column",
            "**FORMAT**: Convert to ISO format YYYY-MM-DD",
            "**DEFAULT**: Return null if no item-specific date exists",
            "**NEGATIVE RULE**: NOT the invoice date, NOT the PO date"
        ]
    ),

    "hsn_code": FieldConfig(
        name="hsn_code",
        display_name="HSN/HS Code",
        field_type=FieldType.STRING,
        description="Harmonized System Code for customs classification",
        extraction_guidelines=[
            "**COLUMN HEADERS**: 'HSN', 'HS Code', 'HSN Code', 'Tariff Code', 'Commodity Code', 'HTS'",
            "**FORMAT**: 6-8 digit numeric code (e.g., '8541.40', '85414000')",
            "**LOCATION**: In line items table or item details section",
            "**VALIDATION**: Should be numeric with optional dots",
            "**DEFAULT**: Return null if not present - DO NOT guess",
            "**NEGATIVE RULE**: NOT a part number, NOT a quantity"
        ]
    ),

    "item_cth": FieldConfig(
        name="item_cth",
        display_name="CTH (Customs Tariff Heading)",
        field_type=FieldType.STRING,
        description="First 4 digits of HS code - Customs Tariff Heading",
        extraction_guidelines=[
            "**DERIVATION**: First 4 digits of HSN code (e.g., HSN 85414000 → CTH 8541)",
            "**COLUMN HEADERS**: 'CTH', 'Tariff Heading', 'Chapter'",
            "**FORMAT**: 4-digit code",
            "**FALLBACK**: If HSN exists, extract first 4 digits",
            "**DEFAULT**: Return null if no CTH or HSN available"
        ]
    ),

    "item_ritc": FieldConfig(
        name="item_ritc",
        display_name="RITC Code",
        field_type=FieldType.STRING,
        description="Regional Import Tariff Code (India-specific)",
        extraction_guidelines=[
            "**COLUMN HEADERS**: 'RITC', 'Import Code'",
            "**FORMAT**: 8-digit code (Indian customs)",
            "**RELATION**: Usually same as or similar to HSN code",
            "**DEFAULT**: Return null if not explicitly provided",
            "**NEGATIVE RULE**: Do NOT copy HSN code unless explicitly labeled RITC"
        ]
    ),

    "item_ceth": FieldConfig(
        name="item_ceth",
        display_name="CETH Code",
        field_type=FieldType.STRING,
        description="Central Excise Tariff Heading (India-specific)",
        extraction_guidelines=[
            "**COLUMN HEADERS**: 'CETH', 'Excise Code', 'Central Excise'",
            "**FORMAT**: 8-digit code (Indian excise)",
            "**DEFAULT**: Return null if not explicitly provided",
            "**NEGATIVE RULE**: Do NOT confuse with HSN or RITC"
        ]
    ),

    "item_quantity": FieldConfig(
        name="item_quantity",
        display_name="Quantity",
        field_type=FieldType.NUMBER,
        description="Number of units ordered",
        required=True,
        extraction_guidelines=[
            "**COLUMN HEADERS**: 'Qty', 'Quantity', 'Ordered', 'Shipped', 'Units', 'Pcs'",
            "**FORMAT**: Numeric value, remove thousand separators",
            "**DECIMALS**: Allow decimals for fractional quantities (kg, meters)",
            "**VALIDATION**: Must be positive number, zero only if explicitly stated",
            "**NEGATIVE RULE**: NOT the unit price, NOT the amount",
            "**NEGATIVE RULE**: If row has no quantity, it's likely a description continuation"
        ]
    ),

    "item_uom": FieldConfig(
        name="item_uom",
        display_name="Unit of Measure",
        field_type=FieldType.STRING,
        description="Unit of measurement for quantity",
        extraction_guidelines=[
            "**COLUMN HEADERS**: 'UOM', 'Unit', 'U/M', 'Measure', 'UM'",
            "**COMMON VALUES**: PCS, EA, NOS, KG, KGS, LBS, MTR, M, CM, MM, LTR, SET, BOX, CTN, PAL",
            "**STANDARDIZATION**: 'Pieces' → 'PCS', 'Each' → 'EA', 'Numbers' → 'NOS', 'Kilograms' → 'KG'",
            "**LOCATION**: Column next to Quantity, or appended to quantity (e.g., '100 PCS')",
            "**FORMAT**: Uppercase abbreviation",
            "**DEFAULT**: Return null if not specified"
        ]
    ),

    "item_unit_price": FieldConfig(
        name="item_unit_price",
        display_name="Unit Price",
        field_type=FieldType.NUMBER,
        description="Price per single unit",
        extraction_guidelines=[
            "**COLUMN HEADERS**: 'Unit Price', 'Price', 'Rate', 'Unit Rate', 'Price/Unit', 'Each'",
            "**FORMAT**: Numeric, remove currency symbols and thousand separators",
            "**DECIMALS**: Preserve decimal precision (up to 4-6 decimal places common)",
            "**CURRENCY**: Use invoice-level currency for all unit prices",
            "**VALIDATION**: If unit_price × quantity ≈ amount, extraction is likely correct",
            "**NEGATIVE RULE**: NOT the total amount, NOT the extended price"
        ]
    ),

    "item_amount": FieldConfig(
        name="item_amount",
        display_name="Line Amount",
        field_type=FieldType.NUMBER,
        description="Total amount for line item (qty × unit price)",
        extraction_guidelines=[
            "**COLUMN HEADERS**: 'Amount', 'Total', 'Extended', 'Line Total', 'Net Amount', 'Value'",
            "**LOCATION**: Usually rightmost column in line items table",
            "**FORMAT**: Numeric, remove currency symbols and thousand separators",
            "**VALIDATION**: Should approximately equal quantity × unit_price",
            "**NEGATIVE RULE**: NOT the invoice grand total, NOT a subtotal row"
        ]
    ),

    "item_origin_country": FieldConfig(
        name="item_origin_country",
        display_name="Country of Origin",
        field_type=FieldType.STRING,
        description="Country where the product was manufactured",
        extraction_guidelines=[
            "**COLUMN HEADERS**: 'Origin', 'Country of Origin', 'COO', 'Made In', 'Manufactured In'",
            "**FORMAT**: Full country name OR 2-letter ISO code (US, CN, JP, DE, SG)",
            "**LOCATION**: In line items table or item details",
            "**PRIORITY**: Prefer explicit column over 'Made in' mentions in description",
            "**DEFAULT**: Return null if not specified per item",
            "**NEGATIVE RULE**: NOT the seller's country, NOT the buyer's country"
        ]
    ),

    "item_mfg_name": FieldConfig(
        name="item_mfg_name",
        display_name="Manufacturer Name",
        field_type=FieldType.STRING,
        description="Name of the product manufacturer",
        extraction_guidelines=[
            "**COLUMN HEADERS**: 'Manufacturer', 'Mfg', 'Maker', 'Brand', 'Producer'",
            "**PRIORITY 1**: Dedicated manufacturer column in line items",
            "**PRIORITY 2**: Manufacturer mentioned in item description",
            "**PRIORITY 3**: If same for all items, may be in document header",
            "**FORMAT**: Full company name",
            "**VENDOR SPECIFIC**: For Commin → Manufacturer is usually 'Murata Electronics'",
            "**DEFAULT**: Return null if not explicitly stated"
        ]
    ),

    "item_mfg_addr": FieldConfig(
        name="item_mfg_addr",
        display_name="Manufacturer Address",
        field_type=FieldType.STRING,
        description="Complete full address of the product manufacturer",
        extraction_guidelines=[
            "**LOCATION**: Usually with manufacturer name, or in separate manufacturer details section",
            "**CRITICAL**: Extract the COMPLETE FULL address if available",
            "**MUST INCLUDE ALL OF THESE (if present)**:",
            "  - Street/Building address",
            "  - City/Town",
            "  - State/Province/Region",
            "  - Postal/ZIP Code",
            "  - Country",
            "**CONCATENATION**: Join multi-line addresses with comma-space",
            "**PRIORITY**: Extract if explicitly provided for line item",
            "**DEFAULT**: Return null if manufacturer address not provided",
            "**NEGATIVE RULE**: NOT the seller address, NOT the buyer address"
        ]
    ),

    "item_mfg_country": FieldConfig(
        name="item_mfg_country",
        display_name="Manufacturer Country",
        field_type=FieldType.STRING,
        description="Country where manufacturer is located",
        extraction_guidelines=[
            "**DERIVATION**: Can be extracted from manufacturer address",
            "**FORMAT**: Full country name OR 2-letter ISO code",
            "**PRIORITY**: If manufacturer address has country, extract it",
            "**RELATION**: Often same as Country of Origin but not always",
            "**DEFAULT**: Return null if not explicitly available"
        ]
    ),
}


AIRWAY_BILL_FIELDS: Dict[str, FieldConfig] = {
    "master_awb_no": FieldConfig(
        name="master_awb_no",
        display_name="Master Air Waybill Number",
        field_type=FieldType.STRING,
        description="Master Air Waybill (MAWB) - main tracking number",
        required=True,
        extraction_guidelines=[
            "**ANCHOR LABELS**: 'MAWB', 'Master AWB', 'Master Air Waybill', 'AWB No', 'Air Waybill'",
            "**FORMAT**: 11 digits, often formatted as XXX-XXXXXXXX (3-digit prefix + 8-digit serial)",
            "**PATTERN**: First 3 digits = airline code (e.g., 176 = Emirates, 057 = Air France)",
            "**CLEANING**: Remove dashes, spaces - extract only digits",
            "**LOCATION**: Usually prominently displayed at top of AWB document",
            "**VALIDATION**: Must be exactly 11 digits after cleaning",
            "**NEGATIVE RULE**: NOT the HAWB, NOT a tracking reference"
        ]
    ),

    "house_awb_no": FieldConfig(
        name="house_awb_no",
        display_name="House Air Waybill Number",
        field_type=FieldType.STRING,
        description="House Air Waybill (HAWB) - freight forwarder reference",
        extraction_guidelines=[
            "**ANCHOR LABELS**: 'HAWB', 'House AWB', 'House Air Waybill', 'HWB', 'House Waybill'",
            "**FORMAT**: Alphanumeric, varies by forwarder (e.g., 'DHL-123456789')",
            "**RELATION**: One MAWB can have multiple HAWBs (consolidated cargo)",
            "**LOCATION**: Usually below or next to MAWB",
            "**DEFAULT**: Return null if document is a Master AWB without House reference",
            "**NEGATIVE RULE**: NOT the MAWB, NOT a booking reference"
        ]
    ),

    "shipper_name": FieldConfig(
        name="shipper_name",
        display_name="Shipper/Sender Name",
        field_type=FieldType.STRING,
        description="Name of the shipping party/sender",
        extraction_guidelines=[
            "**ANCHOR LABELS**: 'Shipper', 'Sender', 'Consignor', 'Ship From', 'From'",
            "**LOCATION**: Usually top-left section of AWB (shipper's name and address block)",
            "**FORMAT**: Full company or person name",
            "**INCLUDE**: Complete legal entity name",
            "**NEGATIVE RULE**: NOT the consignee, NOT the carrier, NOT the agent"
        ]
    ),

    "pkg_in_qty": FieldConfig(
        name="pkg_in_qty",
        display_name="Number of Packages",
        field_type=FieldType.NUMBER,
        description="Total number of packages/pieces in shipment",
        required=True,
        extraction_guidelines=[
            "**ANCHOR LABELS**: 'Pieces', 'Pcs', 'No. of Packages', 'Quantity', 'Total Pieces', 'Number of Pieces'",
            "**FORMAT**: Integer (whole number)",
            "**LOCATION**: In cargo details section, often next to weight",
            "**VALIDATION**: Must be positive integer (1 or more)",
            "**NEGATIVE RULE**: NOT the weight, NOT the carton dimensions"
        ]
    ),

    "gross_weight_kg": FieldConfig(
        name="gross_weight_kg",
        display_name="Gross Weight (KG)",
        field_type=FieldType.NUMBER,
        description="Total gross weight of shipment in kilograms",
        required=True,
        extraction_guidelines=[
            "**ANCHOR LABELS**: 'Gross Weight', 'Weight', 'Actual Weight', 'GW', 'Total Weight'",
            "**FORMAT**: Numeric with decimals, in kilograms",
            "**UNIT CONVERSION**: If in LBS, convert: value × 0.453592 = KG",
            "**LOCATION**: Cargo details section, often in a weight column",
            "**VALIDATION**: Must be positive number",
            "**RELATION**: Gross Weight ≥ Chargeable Weight (usually)",
            "**NEGATIVE RULE**: NOT the volumetric weight, NOT the chargeable weight"
        ]
    ),

    "chargeable_weight_kg": FieldConfig(
        name="chargeable_weight_kg",
        display_name="Chargeable Weight (KG)",
        field_type=FieldType.NUMBER,
        description="Weight used for billing (higher of gross or volumetric)",
        required=True,
        extraction_guidelines=[
            "**ANCHOR LABELS**: 'Chargeable Weight', 'Chrg Wt', 'Billing Weight', 'CW'",
            "**FORMAT**: Numeric with decimals, in kilograms",
            "**CALCULATION**: MAX(Gross Weight, Volumetric Weight)",
            "**VOLUMETRIC**: L×W×H(cm) ÷ 6000 = Volumetric Weight (kg)",
            "**LOCATION**: Near gross weight in cargo details",
            "**VALIDATION**: Must be ≥ Gross Weight (or equal if cargo is dense)",
            "**NEGATIVE RULE**: NOT the gross weight unless they are identical"
        ]
    ),

    "total_frieght": FieldConfig(
        name="total_frieght",
        display_name="Total Freight Charges",
        field_type=FieldType.NUMBER,
        description="Total freight/shipping cost",
        extraction_guidelines=[
            "**ANCHOR LABELS**: 'Freight', 'Total Freight', 'Freight Charges', 'Carriage', 'Shipping Cost'",
            "**FORMAT**: Numeric, remove currency symbols and thousand separators",
            "**LOCATION**: Charges section, usually bottom of AWB",
            "**INCLUDE**: Base freight + surcharges if combined into total",
            "**DEFAULT**: Return null if freight is 'Prepaid' or 'Collect' without amount",
            "**NEGATIVE RULE**: NOT handling fees alone, NOT insurance charges alone"
        ]
    ),

    "frieght_currency": FieldConfig(
        name="frieght_currency",
        display_name="Freight Currency",
        field_type=FieldType.CURRENCY,
        description="Currency of freight charges",
        extraction_guidelines=[
            "**ANCHOR LABELS**: 'Currency', 'Curr', often next to freight amount",
            "**FORMAT**: 3-letter ISO code (USD, EUR, SGD, AED, etc.)",
            "**LOCATION**: Charges section, near total freight",
            "**COMMON VALUES**: USD (most common for air freight), EUR, GBP, AED, SGD",
            "**DEFAULT**: Return null if currency not specified",
            "**NEGATIVE RULE**: NOT the invoice currency (AWB and invoice may differ)"
        ]
    ),
}


# =============================================================================
# 5-WAY VENDOR SPECIFIC OVERRIDES
# =============================================================================

# Helper to clone and update fields
def create_custom_fields(po_override=None, part_override=None, mfg_name_override=None, 
                         mfg_addr_override=None, mfg_country_override=None):
    fields = copy.deepcopy(INVOICE_ITEM_FIELDS)
    if po_override:
        fields["item_po_no"] = po_override
    if part_override:
        fields["item_part_no"] = part_override
    if mfg_name_override:
        fields["item_mfg_name"] = mfg_name_override
    if mfg_addr_override:
        fields["item_mfg_addr"] = mfg_addr_override
    if mfg_country_override:
        fields["item_mfg_country"] = mfg_country_override
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
    name="item_po_no",
    display_name="PO Number (Type 1)",
    field_type=FieldType.STRING,
    description="Order No for Type 1 invoices",
    extraction_guidelines=[
        "**ANCHOR LABEL**: 'ORDER NO:', 'Order No', 'Your Order'",
        "**LOCATION**: Usually in invoice header or per line item",
        "**FORMAT**: Alphanumeric, preserve exact format",
        "**PRIORITY**: If line-level Order No exists, use it; otherwise use header-level",
        "**NEGATIVE RULE**: NOT the invoice number, NOT the article number"
    ]
)
TYPE1_PART = FieldConfig(
    name="item_part_no",
    display_name="Part Number (Type 1)",
    field_type=FieldType.STRING,
    description="Article Number",
    extraction_guidelines=[
        "**STRICT**: Extract from 'Article number' column (e.g. 8940162882)",
        "**NEGATIVE RULE**: Do NOT extract text starting with 'Pack:' or 'Pack: 103...'",
        "Ignore 'NW 104,000'"
    ]
)
TYPE1_FIELDS = create_custom_fields(po_override=TYPE1_PO, part_override=TYPE1_PART)


# --- 3. TYPE 2 (KM_558...) ---
TYPE2_PO = FieldConfig(
    name="item_po_no",
    display_name="PO Number (Type 2)",
    field_type=FieldType.STRING,
    description="Order No for Type 2 invoices",
    extraction_guidelines=[
        "**ANCHOR LABEL**: 'ORDER NO', 'Order No', 'Your Order'",
        "**LOCATION**: Usually in invoice header or per line item",
        "**FORMAT**: Alphanumeric, preserve exact format",
        "**PRIORITY**: If line-level Order No exists, use it; otherwise use header-level",
        "**NEGATIVE RULE**: NOT the Pack number, NOT the Article number"
    ]
)
TYPE2_PART = FieldConfig(
    name="item_part_no",
    display_name="Part Number (Type 2)",
    field_type=FieldType.STRING,
    description="Article Number",
    extraction_guidelines=[
        "**STRICT**: Extract from 'Article number' column (e.g. 4081041590)",
        "**NEGATIVE RULE**: Do NOT extract 'Pack number'"
    ]
)
TYPE2_FIELDS = create_custom_fields(po_override=TYPE2_PO, part_override=TYPE2_PART)


# --- 4. CROWN ---
CROWN_PO = FieldConfig(
    name="item_po_no",
    display_name="PO Number (Crown)",
    field_type=FieldType.STRING,
    description="Customer P.O. for Crown invoices",
    extraction_guidelines=[
        "**ANCHOR LABEL**: 'Customer P.O.', 'Customer PO', 'Your PO'",
        "**LOCATION**: Usually in invoice header section",
        "**FORMAT**: Alphanumeric, preserve exact format",
        "**NEGATIVE RULES**:",
        "  - Ignore 'Order Number' (internal Crown reference)",
        "  - Ignore 'Shipping number'",
        "  - Ignore 'Packing Slip' numbers",
        "  - Ignore 'Invoice Number'"
    ]
)

# Crown Manufacturer - ALWAYS the seller (Crown Germany) for ALL items
CROWN_MFG_NAME = FieldConfig(
    name="item_mfg_name",
    display_name="Manufacturer Name (Crown)",
    field_type=FieldType.STRING,
    description="Manufacturer Name - Always Crown Gabelstapler",
    extraction_guidelines=[
        "**CRITICAL**: For Crown invoices, the manufacturer is ALWAYS the seller",
        "**FIXED VALUE**: 'Crown Gabelstapler GmbH & Co. KG'",
        "**LOGIC**: Crown is the manufacturer/distributor regardless of item's country of origin",
        "**DO NOT**: Use item's 'Made in' country as manufacturer"
    ]
)

CROWN_MFG_ADDR = FieldConfig(
    name="item_mfg_addr",
    display_name="Manufacturer Address (Crown)",
    field_type=FieldType.STRING,
    description="Manufacturer Address - Always Crown's German address",
    extraction_guidelines=[
        "**CRITICAL**: For Crown invoices, use the SELLER'S address as manufacturer address",
        "**FIXED VALUE**: 'Crown Gabelstapler GmbH & Co KG Central Parts Center Parsdorfer Str. 3 85652 Pliening Germany'",
        "**LOGIC**: Extract from invoice header where seller address is printed",
        "**DO NOT**: Use item's 'Made in' country as manufacturer address",
        "**DO NOT**: Extract just the country - get the FULL address"
    ]
)

CROWN_MFG_COUNTRY = FieldConfig(
    name="item_mfg_country",
    display_name="Manufacturer Country (Crown)",
    field_type=FieldType.STRING,
    description="Manufacturer Country - Always Germany",
    extraction_guidelines=[
        "**CRITICAL**: For Crown invoices, manufacturer country is ALWAYS 'Germany'",
        "**LOGIC**: Crown Gabelstapler GmbH & Co. KG is based in Germany",
        "**DO NOT CONFUSE**: item_origin_country (where item was made) ≠ item_mfg_country (where manufacturer is located)",
        "**EXAMPLE**: Item made in USA, but manufacturer (Crown) is in Germany"
    ]
)

CROWN_FIELDS = create_custom_fields(
    po_override=CROWN_PO,
    mfg_name_override=CROWN_MFG_NAME,
    mfg_addr_override=CROWN_MFG_ADDR,
    mfg_country_override=CROWN_MFG_COUNTRY
)

# Remove TOI from Crown Prompt entirely to prevent hallucination
CROWN_HEADER_FIELDS = copy.deepcopy(INVOICE_FIELDS)
if "invoice_toi" in CROWN_HEADER_FIELDS:
    del CROWN_HEADER_FIELDS["invoice_toi"]


# --- 5. ABB / EPIROC ---
ABB_PO = FieldConfig(
    name="item_po_no",
    display_name="PO Number (ABB)",
    field_type=FieldType.STRING,
    description="Order No",
    extraction_guidelines=[
        "**PRIORITY 1**: Extract value from 'Your orderNo.' or 'Our order no'.",
        "**PRIORITY 2**: 'Your reference' (ONLY if alphanumeric).",
        "**NEGATIVE RULE**: If 'Your reference' is a person's name (e.g. 'Imran Shaikh'), IGNORE it."
    ]
)

ABB_PART = FieldConfig(
    name="item_part_no",
    display_name="Part Number (ABB)",
    field_type=FieldType.STRING,
    description="Article Number (Numeric)",
    extraction_guidelines=[
        "**PRIORITY 1**: Look for a NUMERIC article number (can contain spaces, e.g. '9106 0038 59') in the 'Pos Art No' column.",
        "**NEGATIVE RULE**: IGNORE Model Codes or Type IDs (e.g. values starting with '3GAA' or containing text).",
        "**FALLBACK**: If 'Pos Art No' is a Model Code (like '3GAA...'), check the 'Description' column for a numeric Article Number (10+ digits).",
        "**FORMATTING**: Remove ALL spaces from the extracted value."
    ]
)
ABB_FIELDS = create_custom_fields(po_override=ABB_PO, part_override=ABB_PART)


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
            for g in config.extraction_guidelines:
                lines.append(f"    â€¢ {g}")
    return "\n".join(lines)


def generate_extraction_prompt(doc_type, fields, item_fields, custom_instructions=None):
    parts = [f"You are a forensic Customs Data Extractor. Extract data from this {doc_type}.\n"]
    
    if custom_instructions:
        parts.append("### CRITICAL BUSINESS RULES:")
        for rule in custom_instructions:
            parts.append(f"- {rule}")
            
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
You are an Expert Invoice Classification Router. Analyze this multi-page PDF and return a JSON mapping of page ranges to vendor buckets.

IMPORTANT: Return ONLY valid JSON. No explanations, no prose, no markdown code blocks.

================================================================================
SECTION 1: DOCUMENT SPLITTING RULES (Apply First)
================================================================================

A NEW DOCUMENT RANGE starts when ANY of these occur:
1. The INVOICE NUMBER changes
2. The SELLER NAME or LOGO changes
3. Pagination resets (e.g., "Page 1 of 5" appears again)

Each distinct invoice gets its own page range, even if the same vendor type.

================================================================================
SECTION 2: EXCLUSION FILTER (Mandatory Pre-Check)
================================================================================

IMMEDIATELY SKIP pages labeled as:
• Packing List
• Packing Slip  
• Delivery Note
• Purchase Order
• Proforma Invoice
• Quote / Quotation
• Certificate of Origin (standalone)

These pages must NOT appear in any output category. Skip them entirely.

================================================================================
SECTION 3: VENDOR FINGERPRINT MATRIX (Core Classification Logic)
================================================================================

┌─────────────────┬──────────────────────────┬────────────────────────────┬─────────────────────────────────────┐
│ BUCKET          │ KEY ANCHOR (Logo/Seller) │ PRIMARY STRUCTURAL MARKER  │ IDENTIFIER PATTERNS                 │
├─────────────────┼──────────────────────────┼────────────────────────────┼─────────────────────────────────────┤
│ commin_invoices │ Murata / Commin          │ Column: CUSTOMER PART      │ Invoice starts with "SIN-"          │
│                 │                          │ NUMBER                     │                                     │
├─────────────────┼──────────────────────────┼────────────────────────────┼─────────────────────────────────────┤
│ type1_invoices  │ Chicago Pneumatic / AC   │ Column: Article number     │ Invoice starts "0100" or "9002"     │
│                 │                          │                            │ Rows contain "Pack:" text           │
├─────────────────┼──────────────────────────┼────────────────────────────┼─────────────────────────────────────┤
│ type2_invoices  │ Atlas Copco / AC         │ Column: Article number     │ Invoice starts "KM_" or "9002"      │
│                 │                          │                            │ Has dedicated "Pack number" column  │
├─────────────────┼──────────────────────────┼────────────────────────────┼─────────────────────────────────────┤
│ abb_invoices    │ ABB / Epiroc             │ Column: Pos Art No         │ Product codes start with "3GAA"     │
│                 │                          │                            │ or "M3AA"                           │
├─────────────────┼──────────────────────────┼────────────────────────────┼─────────────────────────────────────┤
│ crown_invoices  │ Crown Worldwide          │ Label: Customer P.O.       │ Invoice starts "SHP" or "EXP"       │
├─────────────────┼──────────────────────────┼────────────────────────────┼─────────────────────────────────────┤
│ airway_bills    │ DHL / FedEx / Generic    │ Field: MAWB or HAWB        │ 11-digit AWB numbers                │
│                 │ AWB format               │                            │ Zero unit prices listed             │
├─────────────────┼──────────────────────────┼────────────────────────────┼─────────────────────────────────────┤
│ invoices        │ Generic (Arrow/Siemens/  │ Standard Invoice Header    │ Use when NO specific vendor         │
│ (Global)        │ Other)                   │                            │ fingerprint matches                 │
└─────────────────┴──────────────────────────┴────────────────────────────┴─────────────────────────────────────┘

MATCHING REQUIREMENT: Need 2+ identifiers from the same vendor row to classify as that vendor.

================================================================================
SECTION 4: CONFLICT RESOLUTION LOGIC (Critical Disambiguation)
================================================================================

### THE 9002 TIE-BREAKER (Type 1 vs Type 2)
Both Type 1 and Type 2 can have "9002" prefixes and "Article number" columns.

DISAMBIGUATION RULE:
┌─────────────────────────────────────────────────────────────────────────────┐
│ IF "Pack:" text appears INSIDE description cells → TYPE 1                   │
│ IF "Pack number" is a COLUMN HEADER            → TYPE 2                     │
└─────────────────────────────────────────────────────────────────────────────┘

### THE ABB MODEL FALLBACK
ABB invoices show "Type" or "Model" values (e.g., "3GAA132220-ADE").
RULE: These are MODEL CODES, not part numbers. The actual part number is in "Pos Art No" column.
DO NOT let model codes confuse vendor identification.

### THE GLOBAL DEFAULT RULE
┌─────────────────────────────────────────────────────────────────────────────┐
│ If a document says "INVOICE" but matches ONLY ONE identifier for a         │
│ specific vendor → MUST default to "invoices" (Global bucket)               │
│                                                                             │
│ NEVER classify as a specific vendor with only 1 matching identifier.       │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
SECTION 5: AIRWAY BILL IDENTIFICATION
================================================================================

AIRWAY BILL MARKERS (need 2+ to confirm):
✓ Document header: "AIR WAYBILL" or "AWB" or "MASTER AWB"
✓ Field labeled: "MAWB" (Master Air Waybill) or "HAWB" (House Air Waybill)  
✓ 11-digit tracking number (e.g., "176-17629931")
✓ Weight fields: "Gross Weight", "Chargeable Weight"
✓ NO unit prices for line items (only weights and piece counts)
✓ Carrier branding: DHL, FedEx, UPS, Kuehne+Nagel, etc.

PAGE COUNT: AWBs are almost always 1-2 pages. 3+ pages is unlikely for AWB.

================================================================================
SECTION 6: OUTPUT SPECIFICATION (Strict JSON Only)
================================================================================

Return ONLY this JSON structure. No other text.

{
  "commin_invoices": [[start, end], ...],
  "type1_invoices": [[start, end], ...],
  "type2_invoices": [[start, end], ...],
  "abb_invoices": [[start, end], ...],
  "crown_invoices": [[start, end], ...],
  "invoices": [[start, end], ...],
  "airway_bills": [[start, end], ...]
}

RULES:
• Pages are 1-indexed (first page = 1)
• [5, 10] means pages 5 through 10 inclusive
• Single page = [7, 7]
• Empty category = [] (empty array)
• NO overlapping ranges across categories
• Each page appears in exactly ONE category or is skipped

EXAMPLE OUTPUT:
{
  "commin_invoices": [[1, 2]],
  "type1_invoices": [[3, 25]],
  "type2_invoices": [[26, 28]],
  "abb_invoices": [],
  "crown_invoices": [],
  "invoices": [[29, 30]],
  "airway_bills": [[31, 31]]
}

================================================================================
SECTION 7: DECISION CHECKLIST (Internal Verification)
================================================================================

Before outputting, verify:
□ Did I check for logo/seller name on each document section?
□ Did I identify the column headers in each table?
□ For 9002 invoices: Did I check for "Pack:" text vs "Pack number" column?
□ Does each vendor classification have 2+ matching identifiers?
□ Did I skip all packing lists, delivery notes, and other excluded documents?
□ Are my page ranges non-overlapping and complete?
□ Did I default to "invoices" (global) when uncertain?

NOW ANALYZE THE PDF AND RETURN ONLY THE JSON OUTPUT.
"""

# --- SPECIFIC PROMPTS ---

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
    [
        "**STRICT**: item_po_no = 'ORDER NO'",
        "**STRICT**: item_part_no = 'Article number'",
        "**IGNORE**: 'Pack:' or 'NW' in Article column"
    ]
)

TYPE2_PROMPT = generate_extraction_prompt(
    "Type 2 Invoice",
    INVOICE_FIELDS,
    TYPE2_FIELDS,
    [
        "**STRICT**: item_po_no = 'ORDER NO'",
        "**STRICT**: item_part_no = 'Article number'",
        "**IGNORE**: 'Pack number'"
    ]
)

CROWN_PROMPT = generate_extraction_prompt(
    "Crown Invoice",
    CROWN_HEADER_FIELDS,
    CROWN_FIELDS,
    [
        "**DESCRIPTION CLEANING**:",
        "  - Extract ONLY the product name.",
        "  - The Part Number often appears visually next to the description.",
        "  - CRITICAL: Ensure 'item_description' does NOT contain the 'item_part_no'.",
        "  - If the text implies 'Screw 12345' and PartNo is '12345', extraction must be 'Screw'.",
        "  - STOP reading description at 'Weight:' or 'Made in'.",
        
        "**ORIGIN RULE**:",
        "  - Extract FULL phrase: 'Made in United States'.",
        
        "**HSN/HS CODE RULE**:",
        "  - Look for 'HS-Code', 'Commodity Code', or 'Tariff No' in line items.",
        "  - If not in line items, check footer summaries (often grouped by tax rate).",
        
        "**EXCHANGE RATE RULE**:",
        "  - If the rate is 1.0 or -1.0, return null.",
        "  - Only return extraction if it differs from 1.0.",
        
        "**MANUFACTURER RULE** (CRITICAL):",
        "  - item_mfg_name = ALWAYS 'Crown Gabelstapler GmbH & Co. KG'",
        "  - item_mfg_addr = ALWAYS the seller's full address from invoice header",
        "  - item_mfg_country = ALWAYS 'Germany'",
        "  - DO NOT confuse item_origin_country (Made in USA/Germany/China) with item_mfg_country"
    ]
)

ABB_PROMPT = generate_extraction_prompt(
    "ABB / Epiroc Invoice",
    INVOICE_FIELDS,
    ABB_FIELDS,
    [
        "**PO NUMBER**: Prefer 'Your orderNo.' or 'Our order no'. Ignore names in 'Your reference'.",
        "**PART NUMBER**: Select NUMERIC Article Number (e.g. 9106003859). IGNORE Model Codes (e.g. 3GAA...). Remove spaces.",
        "**FALLBACK**: If 'Pos Art No' is Model Code, look for Numeric Part No in Description."
    ]
)

INVOICE_PROMPT = generate_extraction_prompt(
    "Commercial Invoice",
    INVOICE_FIELDS,
    INVOICE_ITEM_FIELDS,
    [
        "**UNIVERSAL HEURISTIC**: Use visual alignment logic for tables.",
        "**ANCHOR LOGIC**: Logo usually denotes Seller. 'To' usually denotes Buyer.",
        "**MATH CHECK**: Total Amount should equal sum of Line Items (approx).",
        "**DATES**: Standardize all dates to YYYY-MM-DD."
    ]
)

AWB_PROMPT = generate_extraction_prompt(
    "Airway Bill",
    AIRWAY_BILL_FIELDS,
    None,
    ["Extract MAWB (11 digits) and HAWB"]
)


# =============================================================================
# REFINED EXTRACTION CONFIGURATION
# =============================================================================
# These fields are REMOVED from the main prompt and handled via 
# secondary "Text-based verification" to prevent hallucinations.

REFINED_EXTRACTION_CONFIG = {
    "crown_invoice": [
        {
            "field": "invoice_toi",
            "description": "Terms of Invoice / Incoterms. Look for 'Terms of Delivery' label/box in the header.",
            "enabled": True,
            "valid_values": ["EXW", "FCA", "FAS", "FOB", "CFR", "CIF", "CPT", "CIP", "DAP", "DPU", "DDP"]
        }
    ]
}