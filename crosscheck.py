#crosscheck.py: check how many of the names generated is duplicate form thsoe in the training data

import csv
from collections import Counter
import matplotlib.pyplot as plt

# List of CSV file names to combine
csv_files = ["generated_names.csv"] + [f"generated_names_{i}.csv" for i in range(1, 3)]

# Set to hold unique names and a counter for total names (including duplicates)
unique_names = set()
total_names = 0

# Read each CSV file and combine names
for csv_file in csv_files:
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            # Normalize the name by stripping whitespace and converting to lowercase
            name = row[0].strip().lower()
            total_names += 1
            unique_names.add(name)

popular_file = "/clean_data/popularsld.csv"
popular_names = set()

with open(popular_file, "r", newline="", encoding="utf-8") as pf:
    reader = csv.reader(pf)
    for row in reader:
        # Normalize popular names the same way
        pname = row[0].strip().lower()
        popular_names.add(pname)

# Find intersection of unique names and popular names
common_names = unique_names.intersection(popular_names)
print("\nNames found in popular.csv:")
for name in sorted(common_names):
    print(name)

print(f"\n{len(common_names)} out of {len(unique_names)} unique names are found in popular.csv.")
