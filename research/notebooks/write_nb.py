import json
import glob

with open('research/notebooks/extracted_utf8.txt', 'w', encoding='utf-8') as out_f:
    for path in glob.glob('research/notebooks/*.ipynb'):
        out_f.write(f"\n{'='*80}\n")
        out_f.write(f"--- {path} ---\n")
        out_f.write(f"{'='*80}\n")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for cell in data.get('cells', []):
                if cell.get('cell_type') == 'code':
                    source = ''.join(cell.get('source', []))
                    if source.strip():
                        out_f.write(source + "\n")
                        out_f.write("-" * 40 + "\n")
        except Exception as e:
            out_f.write(f"Error reading {path}: {e}\n")
