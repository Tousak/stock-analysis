import json
import glob
import os

for path in glob.glob('research/notebooks/*.ipynb'):
    print(f"\n" + "="*80)
    print(f"--- {path} ---")
    print("="*80)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for cell in data.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                if source.strip():
                    print(source)
                    print("-" * 40)
    except Exception as e:
        print(f"Error reading {path}: {e}")
