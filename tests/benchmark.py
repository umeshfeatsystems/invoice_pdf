import asyncio
import json
import os
import argparse
import sys
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# -----------------------------------------------------------------------------
# 1. SETUP PATHS & IMPORTS (Dynamic & Robust)
# -----------------------------------------------------------------------------
# Get the directory where THIS script is (logistic_pdf/tests)
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (logistic_pdf) to allow imports
ROOT_DIR = os.path.dirname(TESTS_DIR)

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    from services.extractor import extract_invoice, extract_airway_bill
    from models.schemas import Invoice, AirwayBill
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import services. {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

console = Console()

# -----------------------------------------------------------------------------
# 2. CONFIGURATION
# -----------------------------------------------------------------------------
DEFAULT_MODEL = "gemini-3-pro-preview"
DOCS_DIR = os.path.join(TESTS_DIR, "data", "documents")
GT_DIR = os.path.join(TESTS_DIR, "data", "ground_truth")

# Mapping: JSON Key (Left) -> Pydantic Field (Right)
INVOICE_MAPPING = {
    "invoice_number": "invoice_number", 
    "invoice_date": "invoice_date",
    "invoice_currency": "invoice_currency",
    "invoice_toi": "invoice_toi",
    "item_no": "item_no",
    "item_description": "item_description",
    "item_part_no": "item_part_no",
    "item_quantity": "item_quantity",
    "item_unit_price": "item_unit_price",
    "item_origin_country": "item_origin_country",
    "item_mfg_name": "item_mfg_name",
    "item_cth": "item_cth"
}

AWB_MAPPING = {
    "master_awb_no": "master_awb_no",
    "house_awb_no": "house_awb_no", 
    "gross_weight_kg": "gross_weight_kg",
    "chargeable_weight_kg": "chargeable_weight_kg",
    "total_frieght": "total_frieght",
    "frieght_currency": "frieght_currency",
    "pkg_in_qty": "pkg_in_qty"
}

class BenchmarkEngine:
    def __init__(self):
        self.results = []

    def clean_gt_key(self, key: str) -> str:
        """Removes trailing commas and quotes from JSON keys."""
        return key.strip().replace('"', '').replace(',', '').strip()

    def load_ground_truth(self, filename: str) -> list:
        path = os.path.join(GT_DIR, filename)
        if not os.path.exists(path):
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            cleaned_data = []
            for entry in raw_data:
                if not entry: continue
                # Clean every key in the dictionary
                cleaned_entry = {self.clean_gt_key(k): v for k, v in entry.items()}
                cleaned_data.append(cleaned_entry)
            return cleaned_data
        except Exception as e:
            console.print(f"[bold red]Error parsing GT {filename}:[/bold red] {e}")
            return []

    def normalize_val(self, val):
        """Aggressive normalization for comparison."""
        if val is None or val == "null" or val == "": return ""
        s_val = str(val).strip().lower()
        s_val = s_val.replace("made in ", "").replace("\n", " ").replace("  ", " ")
        s_val = s_val.replace(".", "").replace(",", "").replace("-", "") 
        
        # Country codes
        if s_val in ["united states", "usa"]: s_val = "us"
        if s_val in ["germany", "deutschland"]: s_val = "de"
        
        try:
            # Float handling
            f = float(str(val).replace(",", "."))
            if f.is_integer(): return str(int(f))
            return f"{f:.2f}"
        except:
            pass
        return s_val

    def check_match(self, pred, gt, field_name):
        norm_pred = self.normalize_val(pred)
        norm_gt = self.normalize_val(gt)
        
        # 1. Exact Match
        if norm_pred == norm_gt: return True
        
        # 2. Client Req: Description Concatenation Rule
        # If Pred = "Screw 123" and GT = "Screw", it's a match (Pred contains GT)
        if "description" in field_name:
            if norm_gt in norm_pred and len(norm_gt) > 2: return True
        
        # 3. Client Req: FCA -> FOB Transformation
        # If GT says "FCA" but AI output "FOB" (due to business logic), allow it?
        # Actually, if GT is "Target State", it should already be FOB. 
        # If GT is "Raw Source", then we permit mismatch if Pred="FOB" and GT="FCA"
        if "toi" in field_name:
            if "fca" in norm_gt and "fob" in norm_pred: return True

        # 4. Partial Match for long strings
        if len(norm_gt) > 5 and norm_gt in norm_pred: return True
        
        return False

    def find_gt_for_file(self, filename: str, awb_list: list, inv_list: list):
        awb = next((x for x in awb_list if x.get("File Name") == filename), None)
        # Filter strictly for this file
        inv = [x for x in inv_list if x.get("File Name") == filename]
        
        # Fuzzy Fallback
        if not awb and not inv:
            base_name = os.path.splitext(filename)[0]
            if not awb: awb = next((x for x in awb_list if base_name in str(x.get("File Name"))), None)
            if not inv: inv = [x for x in inv_list if base_name in str(x.get("File Name"))]
        
        return awb, inv

    async def run_single_file(self, file_path_input: str):
        # Path Resolution Logic
        target_path = file_path_input
        if not os.path.isabs(file_path_input):
            # Try absolute path from where command was run
            if os.path.exists(file_path_input):
                target_path = os.path.abspath(file_path_input)
            # Try looking in default docs dir
            elif os.path.exists(os.path.join(DOCS_DIR, file_path_input)):
                target_path = os.path.join(DOCS_DIR, file_path_input)
            else:
                console.print(f"[bold red]❌ File not found:[/bold red] {file_path_input}")
                console.print(f"Searched in: {os.getcwd()} and {DOCS_DIR}")
                return

        filename_only = os.path.basename(target_path)
        console.print(Panel(f"[bold yellow]🚀 Benchmarking:[/bold yellow] {filename_only}"))

        # Load GT
        gt_awb_list = self.load_ground_truth("Airwaybill.json")
        gt_inv_list = self.load_ground_truth("InvoiceData.json")
        gt_awb_entry, gt_inv_rows = self.find_gt_for_file(filename_only, gt_awb_list, gt_inv_list)

        if not gt_awb_entry and not gt_inv_rows:
            console.print(f"[bold red]❌ No Ground Truth found in JSON files.[/bold red]")
            return

        # Run Extraction
        if gt_awb_entry:
            await self._benchmark_awb(target_path, gt_awb_entry, filename_only)
        else:
            await self._benchmark_invoice(target_path, gt_inv_rows, filename_only)

    async def _benchmark_awb(self, pdf_path, gt_data, filename):
        try:
            with console.status("[bold green]Gemini 3.0 Pro Thinking...[/bold green]"):
                result, _ = await extract_airway_bill(pdf_path, DEFAULT_MODEL)
            self.print_comparison_table("AWB Results", result.model_dump(), gt_data, AWB_MAPPING, filename, "AWB")
        except Exception as e:
            console.print(f"[bold red]Extraction Failed:[/bold red] {e}")

    async def _benchmark_invoice(self, pdf_path, gt_rows, filename):
        console.print(f"[bold cyan]➤ Mode: Invoice ({len(gt_rows)} Line Items in GT)[/bold cyan]")
        try:
            with console.status("[bold green]Gemini 3.0 Pro Thinking...[/bold green]"):
                result, _ = await extract_invoice(pdf_path, DEFAULT_MODEL)
            
            # Header
            header_gt = gt_rows[0] if gt_rows else {}
            pred_header = result.model_dump(exclude={'items'})
            self.print_comparison_table("Invoice Header", pred_header, header_gt, INVOICE_MAPPING, filename, "Header")

            # Items (Smart Match Logic)
            pred_items = result.items if result.items else []
            console.print(f"\n[dim]Extracted {len(pred_items)} items vs {len(gt_rows)} GT items[/dim]")
            
            matched_indices = set()
            
            for i, gt_item in enumerate(gt_rows):
                best_match = None
                best_idx = -1
                
                # Try to align by Part Number or Description
                gt_part = self.normalize_val(gt_item.get("item_part_no"))
                gt_desc = self.normalize_val(gt_item.get("item_description"))
                
                for p_idx, p_item in enumerate(pred_items):
                    if p_idx in matched_indices: continue
                    
                    p_part = self.normalize_val(p_item.item_part_no)
                    p_desc = self.normalize_val(p_item.item_description)
                    
                    # Match Rule: Part Number exact OR Description contained
                    is_match = False
                    if gt_part and gt_part == p_part: is_match = True
                    elif gt_desc and gt_desc in p_desc: is_match = True
                    
                    if is_match:
                        best_match = p_item
                        best_idx = p_idx
                        break
                
                if best_match:
                    matched_indices.add(best_idx)
                    self.print_comparison_table(f"Line Item {i+1} (Matched with Pred #{best_idx+1})", best_match.model_dump(), gt_item, INVOICE_MAPPING, filename, "Item")
                else:
                    self.print_comparison_table(f"Line Item {i+1} [bold red](NO MATCH FOUND)[/bold red]", {}, gt_item, INVOICE_MAPPING, filename, "Item")

        except Exception as e:
            console.print(f"[bold red]Extraction Failed:[/bold red] {e}")
            import traceback
            traceback.print_exc()

    def print_comparison_table(self, title, pred, gt, mapping, fname, dtype):
        table = Table(title=title, show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Field", style="cyan")
        table.add_column("Ground Truth", style="green")
        table.add_column("AI Prediction", style="yellow")
        table.add_column("Match", justify="center")

        has_data = False
        for gt_key, model_key in mapping.items():
            clean_key = self.clean_gt_key(gt_key)
            gt_val = gt.get(clean_key)
            
            # Skip irrelevant fields
            if "item_" in model_key and dtype == "Header": continue
            if "item_" not in model_key and dtype == "Item": continue
            if gt_val is None: continue
            
            pred_val = pred.get(model_key)
            is_match = self.check_match(pred_val, gt_val, model_key)
            
            self.results.append({
                "File": fname, "Type": dtype, "Field": model_key,
                "GT": gt_val, "Pred": pred_val, "Match": is_match
            })
            
            table.add_row(model_key, str(gt_val)[:50], str(pred_val)[:50], "✅" if is_match else "❌")
            has_data = True
            
        if has_data: console.print(table)

    def save_report(self):
        if self.results:
            df = pd.DataFrame(self.results)
            acc = (df["Match"].sum() / len(df)) * 100
            console.print(Panel(f"[bold white on blue] FINAL ACCURACY: {acc:.2f}% [/bold white on blue]"))
            df.to_csv("benchmark_results.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to PDF file")
    args = parser.parse_args()
    
    engine = BenchmarkEngine()
    asyncio.run(engine.run_single_file(args.file))
    engine.save_report()