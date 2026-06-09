import json

notebook_path = "Phase7.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Notebook keys: {list(nb.keys())}")
if "cells" in nb:
    print(f"Number of cells: {len(nb['cells'])}")
    for i, cell in enumerate(nb["cells"]):
        cell_type = cell.get("cell_type", "unknown")
        source = "".join(cell.get("source", []))
        # print first few lines of source
        lines = source.split("\n")
        first_line = lines[0] if lines else ""
        print(f"Cell {i} [{cell_type}]: {first_line[:80]} ... ({len(lines)} lines)")
