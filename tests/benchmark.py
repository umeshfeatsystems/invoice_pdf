import asyncio
import json
import os
import time
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.status import Status

# -----------------------------------------------------------------------------
# 1. PATH SETUP
# -----------------------------------------------------------------------------
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import extraction logic
try:
    from services.extractor import extract_invoice, extract_airway_bill
except ImportError:
    # Fallback import
    print("⚠️  Importing fallback from main...")
    from main import extract_single_document 
    async def extract_invoice(pdf_path, model):
        return await extract_single_document(pdf_path, "invoice", [1, 100], 1)
    async def extract_airway_bill(pdf_path, model):
        return await extract_single_document(pdf_path, "airway_bill", [1, 100], 1)

console = Console()

# ==========================================
# 👇 CONFIGURATION 👇
# ==========================================
MODEL_TO_USE = "gemini-3-pro-preview" 
DOCS_DIR = os.path.join(current_dir, "data", "documents") 
GT_DIR = os.path.join(current_dir, "data", "ground_truth") 

GT_FILES = {
    "invoice": ["cleaned_0100935473_0302000031.json", "cleanedCROWNInvoice.json", "InvoiceData.json"],
    "airway_bill": ["Airwaybill.json"]
}

class SmartBenchmark:
    def __init__(self):
        self.results = []
        self.gt_id_index = {} 

    def clean_key(self, key: str) -> str:
        return key.strip().replace('"', '').replace(',', '').strip()

    def normalize_id(self, val) -> str:
        return str(val).replace("-", "").replace(" ", "").strip()

    def load_ground_truth(self):
        """Builds a Smart Index of all Ground Truth data."""
        with console.status("[bold yellow]🧠 Building Smart Index from Ground Truth...[/bold yellow]", spinner="dots"):
            count = 0
            for doc_type, files in GT_FILES.items():
                for fname in files:
                    path = os.path.join(GT_DIR, fname)
                    if not os.path.exists(path): continue
                    
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            for entry in data:
                                row = {self.clean_key(k): v for k, v in entry.items()}
                                row["_type"] = doc_type
                                
                                # UNIQUE ID LOGIC
                                unique_id = None
                                if doc_type == "invoice":
                                    unique_id = row.get("invoice_number")
                                else:
                                    unique_id = row.get("master_awb_no")
                                
                                if unique_id:
                                    norm_id = self.normalize_id(unique_id)
                                    if norm_id not in self.gt_id_index:
                                        self.gt_id_index[norm_id] = []
                                    self.gt_id_index[norm_id].append(row)
                                    count += 1
                    except Exception as e:
                        console.print(f"[red]❌ Error loading {fname}: {e}[/red]")

        console.print(f"[green]✅ Indexed {count} rows. Ready to auto-connect documents.[/green]\n")

    def normalize_value(self, val) -> str:
        if val is None: return ""
        s = str(val).strip()
        if s.lower() in ["null", "none", "nan", ""]: return ""
        
        clean = s.replace("€", "").replace("$", "").replace("SGD", "").replace("EUR", "").strip()
        try:
            if "," in clean and "." in clean:
                if clean.find(",") < clean.find("."): clean = clean.replace(",", "") 
                else: clean = clean.replace(".", "").replace(",", ".") 
            elif "," in clean: clean = clean.replace(",", ".") 
            
            f = float(clean)
            if f.is_integer(): return str(int(f))
            return f"{f:.2f}"
        except:
            pass
        return s.lower().replace("\n", " ").replace("  ", " ").strip()

    def flatten_prediction(self, result, doc_type):
        rows = []
        data = result.data if hasattr(result, "data") else result
        if hasattr(data, "model_dump"): data = data.model_dump()
        if not data: return []

        if doc_type == "invoice":
            header = {k: v for k, v in data.items() if k != "items"}
            items = data.get("items", []) or []
            if not items: rows.append(header)
            else:
                for item in items:
                    row = header.copy()
                    if hasattr(item, "model_dump"): item = item.model_dump()
                    row.update(item)
                    rows.append(row)
        else:
            rows.append(data)
        return rows

    async def run(self):
        self.load_ground_truth()
        
        if not os.path.exists(DOCS_DIR):
             console.print(f"[bold red]❌ Directory not found: {DOCS_DIR}[/bold red]")
             return

        pdf_files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith('.pdf')]
        console.print(f"[bold cyan]🚀 Found {len(pdf_files)} PDFs. Starting Auto-Analysis...[/bold cyan]\n")

        for filename in pdf_files:
            pdf_path = os.path.join(DOCS_DIR, filename)
            
            console.rule(f"[bold]Processing: {filename}[/bold]")
            start_time = time.time()
            
            extracted_data = None
            doc_type = "invoice"
            if "awb" in filename.lower(): doc_type = "airway_bill"
            
            # --- 1. LIVE EXTRACTION STATUS ---
            try:
                # This spinner stays active while extract_invoice runs
                with console.status(f"[bold blue]🤖 Gemini is extracting {doc_type}...[/bold blue]", spinner="dots") as status:
                    if doc_type == "airway_bill": 
                        res = await extract_airway_bill(pdf_path, MODEL_TO_USE)
                    else:
                        res = await extract_invoice(pdf_path, MODEL_TO_USE)
                    
                    status.update(f"[bold green]Extraction complete![/bold green]")
                    time.sleep(0.5) # Brief pause to show green
                
                if isinstance(res, tuple): res = res[0]
                extracted_data = res
                
            except Exception as e:
                console.print(f"[red]💥 Extraction Failed:[/red] {e}")
                continue

            duration = time.time() - start_time
            
            # --- 2. AUTO-CONNECT ---
            data_dict = extracted_data.data if hasattr(extracted_data, "data") else extracted_data
            if hasattr(data_dict, "model_dump"): data_dict = data_dict.model_dump()
            
            extracted_id = str(data_dict.get("invoice_number" if doc_type == "invoice" else "master_awb_no", ""))
            norm_id = self.normalize_id(extracted_id)
            gt_rows = self.gt_id_index.get(norm_id)
            
            if not gt_rows:
                console.print(f"   [yellow]⚠️  Extracted ID: [bold]{extracted_id}[/bold] (Not found in Ground Truth JSONs)[/yellow]")
                continue
            
            console.print(f"   [bold green]🔗 Matched GT for ID: {extracted_id}[/bold green]")
            
            # --- 3. COMPARE ---
            pred_rows = self.flatten_prediction(extracted_data, doc_type)
            max_rows = max(len(gt_rows), len(pred_rows))
            console.print(f"   ⏱️  Time: {duration:.2f}s | AI Found {len(pred_rows)} items vs GT {len(gt_rows)} items")

            KEY_MAP = {
                "invoice_number": "invoice_number", "master_awb_no": "master_awb_no",
                "house_awb_no": "house_awb_no", "gross_weight_kg": "gross_weight_kg",
                "item_part_no": "item_part_no", "item_description": "item_description",
                "item_quantity": "item_quantity", "item_unit_price": "item_unit_price",
                "invoice_date": "invoice_date", "item_amount": "item_amount"
            }
            SKIP_KEYS = ["File Name", "Page range", "_type", "item_amt"]

            matches_found = 0
            for i in range(max_rows):
                gt_row = gt_rows[i] if i < len(gt_rows) else {}
                pred_row = pred_rows[i] if i < len(pred_rows) else {}
                
                for k, gt_val in gt_row.items():
                    if k in SKIP_KEYS: continue
                    model_key = KEY_MAP.get(k, k)
                    pred_val = pred_row.get(model_key)
                    
                    match = self.normalize_value(gt_val) == self.normalize_value(pred_val)
                    if match: matches_found += 1
                    
                    self.results.append({
                        "File": filename, "Field": k, "GT": str(gt_val), 
                        "AI": str(pred_val), "Match": match
                    })
            
            console.print(f"   ✅ Validated {matches_found} fields.")

        self.print_report()

    def print_report(self):
        if not self.results: return
        df = pd.DataFrame(self.results)
        acc = (df["Match"].sum() / len(df)) * 100 if len(df) > 0 else 0
        
        failures = df[~df["Match"]]
        
        console.print("\n")
        table = Table(title=f"Benchmark Report (Accuracy: {acc:.2f}%)")
        table.add_column("File", style="cyan")
        table.add_column("Field", style="magenta")
        table.add_column("Expected", style="green")
        table.add_column("Actual", style="red")
        
        for _, row in failures.head(15).iterrows():
            table.add_row(row["File"], row["Field"], row["GT"][:40], row["AI"][:40])
            
        console.print(table)
        if len(failures) > 15: console.print(f"[yellow]...and {len(failures)-15} more errors.[/yellow]")
        
        df.to_csv("benchmark_results.csv", index=False)
        console.print(f"\n[bold green]✅ Report saved to benchmark_results.csv[/bold green]")

if __name__ == "__main__":
    asyncio.run(SmartBenchmark().run())