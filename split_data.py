
import pandas as pd
from pathlib import Path

# Parameters
DATA_DIR = Path('data')
OUTPUT_DIR = DATA_DIR / 'split'
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # Remaining 15%

def split_data():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created: {OUTPUT_DIR}")
    
    # Process each CSV file in data directory
    csv_files = list(DATA_DIR.glob('*.csv'))
    
    if not csv_files:
        print("No CSV files found in data directory.")
        return

    print(f"Found {len(csv_files)} files to split.\n")
    
    for file_path in csv_files:
        filename = file_path.name
        stem = file_path.stem  # e.g., 'ETTh1'
        
        print(f"[{stem}] Processing...")
        
        # Load data
        df = pd.read_csv(file_path)
        n = len(df)
        
        # Calculate split indices
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        
        # Split data
        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]
        
        # Save splits
        train_path = OUTPUT_DIR / f'{stem}_train.csv'
        val_path = OUTPUT_DIR / f'{stem}_val.csv'
        test_path = OUTPUT_DIR / f'{stem}_test.csv'
        
        train.to_csv(train_path, index=False)
        val.to_csv(val_path, index=False)
        test.to_csv(test_path, index=False)
        
        print(f"  Total rows: {n}")
        print(f"  Train: {len(train)} ({len(train)/n:.1%}) -> {train_path.name}")
        print(f"  Val:   {len(val)} ({len(val)/n:.1%}) -> {val_path.name}")
        print(f"  Test:  {len(test)} ({len(test)/n:.1%}) -> {test_path.name}")
        print("  ✓ Saved successfully\n")

if __name__ == "__main__":
    split_data()
