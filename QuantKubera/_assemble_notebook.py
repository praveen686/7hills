#!/usr/bin/env python3
"""Assemble the enhanced QuantKubera Monolith v2 notebook from cell files."""
import json
import os
import uuid

def code_cell(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": src.strip()
    }

def md_cell(src):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": src.strip()
    }

def read_cell_file(path):
    with open(path, 'r') as f:
        return f.read()

cells = []

# Cell 0: Markdown title
cells.append(md_cell(read_cell_file('_cells/cell0_markdown.md')))

# Code cells (ordered)
cell_files = [
    '_cells/cell1.py',   # Setup & Configuration
    '_cells/cell2.py',   # Data Engine — KiteAuth + KiteFetcher
    '_cells/cell2b.py',  # Cross-Asset Data — News Sentiment + India VIX
    '_cells/cell3.py',   # Feature Engineering (31 features)
    '_cells/cell4.py',   # AFML Pipeline
    '_cells/cell5.py',   # MomentumTransformer + SharpeLoss
    '_cells/cell6.py',   # Walk-Forward Validation Engine
    '_cells/cell7.py',   # Orchestration, Metrics & Visualization
    '_cells/cell8.py',   # VectorBTPro Tearsheet
]
for path in cell_files:
    cells.append(code_cell(read_cell_file(path)))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "qk_venv",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "QuantKubera_Monolith_v2.ipynb")
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Generated {out_path} with {len(cells)} cells")
