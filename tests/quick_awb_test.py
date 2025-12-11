"""Quick AWB-only accuracy test."""
import asyncio
import os
import sys
sys.path.append('..')

from services.extractor import extract_airway_bill

# Ground truth for 2 AWB files
GT = [
    {
        "File Name": "HAWB_SIN-10076768_SGSIN261120252025112618251148799999AFHAWB.pdf",
        "master_awb_no": "17617629931",
        "house_awb_no": "SIN10076768",
        "pkg_in_qty": 2,
        "gross_weight_kg": 176,
        "chargeable_weight_kg": 300,
        "total_frieght": 850,
        "frieght_currency": "SGD"
    },
    {
        "File Name": "neb06511186.pdf",
        "master_awb_no": "29760626370",
        "house_awb_no": "NEB06511186",
        "pkg_in_qty": 11,
        "gross_weight_kg": 905.6,
        "chargeable_weight_kg": 1103,
        "frieght_currency": "EUR"
    },
]

async def test_awbs():
    total_matches = 0
    total_fields = 0
    
    for gt in GT:
        pdf_path = os.path.join("data", "documents", gt["File Name"])
        if not os.path.exists(pdf_path):
            print(f"SKIP: {gt['File Name']} not found")
            continue
            
        print(f"\n=== Testing: {gt['File Name']} ===")
        result, _ = await extract_airway_bill(pdf_path, "gemini-2.5-flash")
        r = result.model_dump()
        
        for k, v in gt.items():
            if k == "File Name":
                continue
            total_fields += 1
            pred = r.get(k)
            
            # Normalize for comparison
            str_pred = str(pred).strip() if pred else ""
            str_truth = str(v).strip()
            
            # Try numeric comparison first
            match = False
            try:
                if float(str_pred) == float(str_truth):
                    match = True
            except (ValueError, TypeError):
                match = (str_pred == str_truth)
            
            if match:
                total_matches += 1
                print(f"  OK: {k} = {v}")
            else:
                print(f"  MISS: {k}")
                print(f"        Expected: {v}")
                print(f"        Got:      {pred}")
    
    print(f"\n=== OVERALL: {total_matches}/{total_fields} ({100*total_matches/total_fields:.1f}%) ===")

if __name__ == "__main__":
    asyncio.run(test_awbs())
