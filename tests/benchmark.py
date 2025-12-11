import asyncio
import json
import os
import time
import argparse
import pandas as pd
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

# Add parent dir to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.extractor import extract_invoice, extract_airway_bill
from models.schemas import Invoice, AirwayBill

console = Console()

# ==========================================
# 👇 CONFIGURATION 👇
# ==========================================
DEFAULT_TEST_FILE = "neb06511186.pdf" 
MODEL_TO_USE = "gemini-3-pro-preview" 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "data", "documents")
GT_DIR = os.path.join(BASE_DIR, "data", "ground_truth")

# --- MAPPINGS ---
AWB_MAPPING = {
    "master_awb_no": "Master AWB",
    "house_awb_no": "House AWB", 
    "gross_weight_kg": "Gross Wt (KG)",
    "chargeable_weight_kg": "Chg Wt (KG)",
    "total_frieght": "Total Freight",
    "frieght_currency": "Currency",
    "pkg_in_qty": "Pkg Qty"
}

INVOICE_MAPPING = {
    "invoice_number": "Inv #",
    "invoice_date": "Inv Date",
    "invoice_currency": "Currency",
    "invoice_toi": "IncoTerms",
    "item_no": "Item #",
    "item_description": "Description",
    "item_part_no": "Part No",
    "item_quantity": "Qty",
    "item_unit_price": "Unit Price",
    "item_origin_country": "COO"
}

class BenchmarkEngine:
    def __init__(self):
        self.results = []

    def clean_gt_key(self, key: str) -> str:
        return key.strip().replace(",", "").replace('"', '').strip()

    def load_ground_truth(self, filename: str) -> List[Dict]:
        path = os.path.join(GT_DIR, filename)
        if not os.path.exists(path):
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            return [{self.clean_gt_key(k): v for k, v in entry.items()} for entry in raw_data if entry]
        except: return []

    def flatten_invoice_extraction(self, invoice: Invoice, filename: str) -> List[Dict]:
        flat_rows = []
        invoice_dict = invoice.model_dump()
        header_data = {k: v for k, v in invoice_dict.items() if k != "items"}
        
        if not invoice.items:
            header_data["File Name"] = filename
            flat_rows.append(header_data)
            return flat_rows

        for item in invoice.items:
            row = header_data.copy()
            row.update(item.model_dump())
            row["File Name"] = filename
            flat_rows.append(row)
        return flat_rows

    def smart_normalize(self, val: Any) -> str:
        if val is None or val == "null" or val == "": return ""
        s_val = str(val).strip()
        if len(s_val) >= 10 and s_val[4] == "-" and s_val[7] == "-": return s_val[:10]
        if s_val.replace('.', '', 1).isdigit():
             try:
                 f = float(s_val)
                 if f.is_integer(): return str(int(f))
                 return f"{f:.2f}"
             except: pass
        return s_val.lower().replace("made in ", "").replace("\n", " ").replace("  ", " ")

    def check_accuracy(self, pred: Any, truth: Any) -> bool:
        return self.smart_normalize(pred) == self.smart_normalize(truth)

    async def run_benchmark(self, target_file: str):
        console.print(Panel.fit(f"[bold yellow]🚀 Starting Benchmark: {target_file}[/bold yellow]", border_style="yellow"))

        gt_awb = self.load_ground_truth("Airwaybill.json")
        gt_inv = self.load_ground_truth("InvoiceData.json")
        
        file_gt_awb = next((x for x in gt_awb if x.get("File Name") == target_file), None)
        file_gt_inv_rows = [x for x in gt_inv if x.get("File Name") == target_file]

        if not file_gt_awb and not file_gt_inv_rows:
            console.print(f"[bold red]❌ No Ground Truth found for {target_file}[/bold red]")
            return

        pdf_path = os.path.join(DOCS_DIR, target_file)
        if not os.path.exists(pdf_path):
            console.print(f"[bold red]❌ PDF not found: {pdf_path}[/bold red]")
            return

        # --- AWB PROCESSING ---
        if file_gt_awb:
            console.print("\n[bold cyan]➤ Processing Airway Bill...[/bold cyan]")
            
            start_time = time.time()
            extracted_awb = None
            
            # Live Spinner for "Thinking" phase
            with console.status("[bold green]Gemini 3.0 Pro is thinking (Reasoning Phase)...[/bold green]", spinner="dots"):
                try:
                    extracted_awb, _ = await extract_airway_bill(pdf_path, MODEL_TO_USE)
                except Exception as e:
                    console.print(f"[bold red]❌ Error:[/bold red] {e}")

            duration = time.time() - start_time
            console.print(f"[dim]Finished in {duration:.2f}s[/dim]")

            if extracted_awb:
                self.display_live_comparison("AWB Results", extracted_awb.model_dump(), file_gt_awb, AWB_MAPPING, target_file, "AWB")

        # --- INVOICE PROCESSING ---
        if file_gt_inv_rows:
            console.print("\n[bold cyan]➤ Processing Invoice...[/bold cyan]")
            
            start_time = time.time()
            extracted_inv = None
            
            with console.status("[bold green]Gemini 3.0 Pro is thinking (Reasoning Phase)...[/bold green]", spinner="dots"):
                try:
                    extracted_inv, _ = await extract_invoice(pdf_path, MODEL_TO_USE)
                except Exception as e:
                    console.print(f"[bold red]❌ Error:[/bold red] {e}")

            duration = time.time() - start_time
            console.print(f"[dim]Finished in {duration:.2f}s[/dim]")

            if extracted_inv:
                flat_preds = self.flatten_invoice_extraction(extracted_inv, target_file)
                # Compare Row-by-Row
                max_len = max(len(file_gt_inv_rows), len(flat_preds))
                
                for i in range(max_len):
                    gt_row = file_gt_inv_rows[i] if i < len(file_gt_inv_rows) else {}
                    pred_row = flat_preds[i] if i < len(flat_preds) else {}
                    
                    console.print(f"\n[bold underline]Line Item {i+1}[/bold underline]")
                    # Mapping generic invoice keys to internal keys
                    # Using a subset map for display to fit screen
                    DISPLAY_MAP = {k: v for k, v in INVOICE_MAPPING.items()}
                    
                    self.display_live_comparison(f"Row {i+1}", pred_row, gt_row, DISPLAY_MAP, target_file, "Invoice", row_idx=i+1)

    def display_live_comparison(self, title, pred_data, gt_data, mapping, filename, doc_type, row_idx=None):
        """Prints a beautiful table of Ground Truth vs Predicted immediately."""
        table = Table(title=title, show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Ground Truth", style="green")
        table.add_column("AI Prediction", style="yellow")
        table.add_column("Status", justify="center")

        for gt_key, label in mapping.items():
            # Only show if GT exists for this field
            if gt_key in gt_data:
                # Find matching model key (reverse map check)
                model_key = gt_key.strip() # Assuming simplistic 1-to-1 for this display
                
                # Try to find the value in prediction using mapped keys from benchmark config
                # Re-using the INVOICE_MAPPING / AWB_MAPPING from global scope logic
                # We need to map 'gt_key' -> 'model_key'
                
                # Reverse lookup or direct usage? 
                # In this script, keys in 'mapping' dict are GT keys.
                # We need to know which model field corresponds to it.
                # For simplicity in this display function, we try direct match or mapped match.
                
                # Fix: Use the global mappings to find the Pydantic key
                pydantic_key = None
                if doc_type == "AWB":
                    pydantic_key = gt_key # AWB Mapping keys are identical in this script's config
                else:
                    pydantic_key = gt_key.strip() # Invoice keys

                t_val = gt_data.get(gt_key)
                p_val = pred_data.get(pydantic_key)
                
                is_match = self.check_accuracy(p_val, t_val)
                status = "✅" if is_match else "❌"
                
                # Record result
                self.results.append({
                    "File": filename, "Type": doc_type, "Field": label,
                    "Ground Truth": t_val, "Predicted": p_val, "Match": is_match
                })

                table.add_row(
                    label,
                    str(t_val)[:50],
                    str(p_val)[:50],
                    status
                )

        console.print(table)

    def print_report(self):
        if not self.results: return
        df = pd.DataFrame(self.results)
        acc = (df["Match"].sum() / len(df)) * 100
        console.print(f"\n[bold white on blue] FINAL ACCURACY: {acc:.2f}% [/bold white on blue]\n")
        df.to_csv("benchmark_results.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=DEFAULT_TEST_FILE)
    args = parser.parse_args()
    
    target = args.file if args.file else DEFAULT_TEST_FILE
    engine = BenchmarkEngine()
    asyncio.run(engine.run_benchmark(target_file=target))
    engine.print_report()