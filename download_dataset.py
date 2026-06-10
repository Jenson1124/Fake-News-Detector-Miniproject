"""
Download the Fake News dataset from Kaggle.

Option A (recommended): Kaggle CLI
  pip install kaggle
  # Put your kaggle.json in ~/.kaggle/
  kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset
  unzip fake-and-real-news-dataset.zip -d dataset/

Option B: Manual download
  1. Go to: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
  2. Download and unzip into a folder called  dataset/
  3. You should have:
       dataset/True.csv
       dataset/Fake.csv

Option C: Quick sample for testing (run this script directly)
  python download_dataset.py --sample
  This creates small sample CSVs so you can test the pipeline immediately.
"""

import argparse
import os
import pandas as pd

SAMPLE_REAL = [
    ("WASHINGTON", "politics", "Reuters", "2017-01-01",
     "The White House said on Monday that President Trump signed an executive order."),
    ("NEW YORK", "politics", "Reuters", "2017-02-10",
     "The United States Senate voted to confirm the new cabinet secretary after a lengthy debate."),
    ("WASHINGTON", "politics", "Reuters", "2017-03-15",
     "Congress approved a bipartisan spending bill to keep the government funded through the fiscal year."),
    ("LONDON", "worldnews", "Reuters", "2017-04-20",
     "British Prime Minister Theresa May announced plans for a snap general election in June."),
    ("PARIS", "worldnews", "Reuters", "2017-05-08",
     "Emmanuel Macron won the French presidential election defeating Marine Le Pen."),
    ("BERLIN", "worldnews", "Reuters", "2017-06-12",
     "German Chancellor Angela Merkel met with world leaders at the G20 summit in Hamburg."),
    ("BEIJING", "worldnews", "Reuters", "2017-07-01",
     "China celebrated the 20th anniversary of Hong Kong's return to Chinese sovereignty."),
    ("MOSCOW", "worldnews", "Reuters", "2017-08-14",
     "Russia's foreign minister held talks with US Secretary of State over Syria ceasefire."),
    ("TOKYO", "worldnews", "Reuters", "2017-09-22",
     "Japan's Prime Minister called a snap election amid rising tensions with North Korea."),
    ("OTTAWA", "worldnews", "Reuters", "2017-10-05",
     "Canada and the United States resumed NAFTA renegotiation talks in Washington."),
]

SAMPLE_FAKE = [
    ("", "News", "", "December 31, 2017",
     "BREAKING: Deep state operatives have been secretly controlling the White House for decades."),
    ("", "News", "", "January 15, 2018",
     "Scientists CONFIRM that vaccines contain microchips to track the population worldwide."),
    ("", "politics", "", "February 3, 2018",
     "SHOCK REPORT: The mainstream media is hiding the truth about the government's secret plans."),
    ("", "News", "", "March 22, 2018",
     "EXPOSED: Globalists plan to replace the dollar with a new world currency by end of year."),
    ("", "left-news", "", "April 11, 2018",
     "Anonymous sources reveal that shadowy figures control all major elections using rigged machines."),
    ("", "News", "", "May 5, 2018",
     "HUGE: The FBI has been covering up evidence that would expose the entire corrupt establishment."),
    ("", "politics", "", "June 18, 2018",
     "BOMBSHELL: Secret documents prove the moon landing was staged in a Hollywood studio."),
    ("", "News", "", "July 4, 2018",
     "ALERT: Government is secretly spraying chemicals from planes to control the population's minds."),
    ("", "News", "", "August 9, 2018",
     "REVEALED: Major pharmaceutical companies deliberately cause diseases to sell more medicine."),
    ("", "left-news", "", "September 30, 2018",
     "INSIDER LEAK: The entire electoral college has been compromised by foreign operatives."),
]

def create_sample():
    os.makedirs("dataset", exist_ok=True)
    real_df = pd.DataFrame(SAMPLE_REAL, columns=["subject", "title", "author", "date", "text"])
    fake_df = pd.DataFrame(SAMPLE_FAKE, columns=["subject", "title", "author", "date", "text"])
    real_df.to_csv("dataset/True.csv", index=False)
    fake_df.to_csv("dataset/Fake.csv", index=False)
    print("✅ Sample dataset created in dataset/")
    print(f"   True.csv  : {len(real_df)} rows")
    print(f"   Fake.csv  : {len(fake_df)} rows")
    print("\n⚠️  This is a tiny sample for testing only.")
    print("   For your final project, download the full Kaggle dataset (44,000+ articles).")
    print("   See the instructions at the top of this file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Create a small sample dataset for testing")
    args = parser.parse_args()
    if args.sample:
        create_sample()
    else:
        print(__doc__)
