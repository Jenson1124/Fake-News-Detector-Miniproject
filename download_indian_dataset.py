"""
download_indian_dataset.py
==========================
Instructions and helper for getting the Indian fake news dataset.

OPTION 1 — Direct Kaggle download (recommended):
    pip install kaggle
    kaggle datasets download -d vikasukani/fake-news-detection-dataset-in-india
    unzip fake-news-detection-dataset-in-india.zip -d dataset/indian

OPTION 2 — Manual download:
    1. Go to: https://www.kaggle.com/datasets/vikasukani/fake-news-detection-dataset-in-india
    2. Click Download
    3. Extract and place CSV files into: dataset/indian/

OPTION 3 — Alternate Indian dataset (more articles):
    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    (search "fake news india" on Kaggle for more options)

After placing files in dataset/indian/, retrain:
    python train_model.py

The model will automatically detect and combine:
    - US news: dataset/True.csv + dataset/Fake.csv
    - Indian news: dataset/indian/*.csv
"""

import os, sys

def check_status():
    india_dir = 'dataset/indian'
    print("\n📁 Dataset status check")
    print("─" * 40)

    us_true = os.path.exists('dataset/True.csv')
    us_fake = os.path.exists('dataset/Fake.csv')
    print(f"  US True.csv  : {'✅ Found' if us_true else '❌ Missing'}")
    print(f"  US Fake.csv  : {'✅ Found' if us_fake else '❌ Missing'}")

    if os.path.isdir(india_dir):
        csvs = [f for f in os.listdir(india_dir) if f.endswith('.csv')]
        if csvs:
            print(f"  Indian CSVs  : ✅ Found {len(csvs)} file(s): {', '.join(csvs)}")
        else:
            print(f"  Indian CSVs  : ❌ Folder exists but empty — place CSV files in dataset/indian/")
    else:
        print(f"  Indian CSVs  : ❌ dataset/indian/ folder missing")
        print(f"\n  To add Indian data:")
        print(f"    mkdir -p dataset/indian")
        print(f"    # Download from Kaggle and place CSV files there")

    print()

if __name__ == '__main__':
    check_status()
    print(__doc__)
