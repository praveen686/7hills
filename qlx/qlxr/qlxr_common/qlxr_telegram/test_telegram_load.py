
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brahmastra.datafetch.external_csv import process_external_csv

def main():
    # Sample file from the extracted telegram data
    sample_file = "data/raw/telegram/2021-12-09/BANKNIFTY36000CE 2021-12-09.csv"
    
    if not Path(sample_file).exists():
        print(f"Error: Sample file not found at {sample_file}")
        return

    print(f"Processing Telegram Sample: {sample_file}")
    df = process_external_csv(sample_file, source='telegram')
    
    print("\nNormalized Telegram Data (Tail):")
    print(df.tail())
    print("\nColumns:", df.columns.tolist())
    print(f"Dataframe Shape: {df.shape}")

if __name__ == "__main__":
    main()
