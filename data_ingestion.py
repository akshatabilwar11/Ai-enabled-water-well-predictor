import os
import json
import requests
import pdfplumber
import pandas as pd

CGWB_URL = "https://cgwb.gov.in/sites/default/files/inline-files/january_wl_1994-2024-compressed.pdf"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PDF_PATH = os.path.join(DATA_DIR, "cgwb_january_wl_1994_2024.pdf")
CSV_PATH = os.path.join(DATA_DIR, "cgwb_tables.csv")
SUMMARY_PATH = os.path.join(DATA_DIR, "cgwb_summary.json")


def ensure_data_dir():
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


def download_pdf(url: str = CGWB_URL, dest: str = PDF_PATH) -> str:
    ensure_data_dir()
    if os.path.isfile(dest):
        return dest
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)
    return dest


def extract_tables_to_csv(pdf_path: str = PDF_PATH, csv_path: str = CSV_PATH, summary_path: str = SUMMARY_PATH) -> str:
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            for t_idx, table in enumerate(tables, start=1):
                if not table or len(table) < 2:
                    continue
                header = table[0]
                # Normalize header names
                header = [str(h).strip().lower().replace("\n", " ") if h is not None else f"col_{i}" for i, h in enumerate(header)]
                for row in table[1:]:
                    record = {header[i] if i < len(header) else f"col_{i}": (str(val).strip() if val is not None else None) for i, val in enumerate(row)}
                    record["_page"] = page_idx
                    record["_table"] = t_idx
                    rows.append(record)

    if not rows:
        # Create empty CSV
        pd.DataFrame([]).to_csv(csv_path, index=False)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"pages": 0, "tables": 0, "rows": 0}, f, indent=2)
        return csv_path

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # Basic summary
    summary = {
        "pages": df["_page"].max() if "_page" in df.columns else None,
        "tables": int(df["_table"].max()) if "_table" in df.columns else None,
        "rows": int(len(df)),
        "columns": list(df.columns),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return csv_path


def main():
    pdf = download_pdf()
    csv = extract_tables_to_csv(pdf)
    print(f"Saved CSV: {csv}")
    print(f"Saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()


