# analysis.py: fidn the % duplicates in generated names; append TLDs to distinct names and save to a txt file

import csv
from collections import Counter
import matplotlib.pyplot as plt

# List of CSV file names to combine
csv_files = ["generated_names.csv"] + [f"generated_names_{i}.csv" for i in range(1, 3)]

#csv_files = [f"generated_names_{i}.csv" for i in range(1, 6)]

# Set to hold unique names and a counter for total names (including duplicates)
unique_names = set()
total_names = 0

# Read each CSV file and combine names
for csv_file in csv_files:
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            # Normalize the name (strip whitespace and convert to lowercase)
            name = row[0].strip().lower()
            total_names += 1
            unique_names.add(name)

# Calculate duplicates and percentage of duplicate entries
distinct_count = len(unique_names)
duplicate_count = total_names - distinct_count
duplicate_percentage = (duplicate_count / total_names) * 100

print(f"Total names (including duplicates): {total_names}")
print(f"Distinct names: {distinct_count}")
print(f"Percentage duplicates: {duplicate_percentage:.2f}%")

lengths = [len(name) for name in unique_names]
average_length = sum(lengths) / len(lengths) if lengths else 0
min_length = min(lengths) if lengths else 0
max_length = max(lengths) if lengths else 0

print("\nName Length Statistics:")
print(f"Average length: {average_length:.2f}")
print(f"Minimum length: {min_length}")
print(f"Maximum length: {max_length}")

# Compute the overall character distribution across all unique names
char_distribution = Counter()
for name in unique_names:
    char_distribution.update(name)

print("\nCharacter Distribution:")
for char in sorted(char_distribution):
    print(f"{char}: {char_distribution[char]}")

# Plotting the name length distribution
plt.figure(figsize=(14, 6))

# Subplot 1: Histogram for name lengths
plt.subplot(1, 2, 1)
plt.hist(lengths, bins=range(min_length, max_length + 2), edgecolor='black', alpha=0.7)
plt.title('Distribution of Name Lengths')
plt.xlabel('Name Length')
plt.ylabel('Frequency')

# Subplot 2: Bar chart for character distribution
plt.subplot(1, 2, 2)
# Sort characters alphabetically for plotting
chars = sorted(char_distribution.keys())
frequencies = [char_distribution[char] for char in chars]
plt.bar(chars, frequencies, color='skyblue', edgecolor='black')
plt.title('Character Frequency Distribution')
plt.xlabel('Character')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig("plot.png")
plt.show()

# List of TLDs to add to the names
tlds = [
    "com", "net", "de", "jp", "org", "cn", "io",
    "pw", "it", "br", "ru", "fr", "uk", "nl",
    "pl", "link", "au", "za", "info", "xyz"
]

# Create a list of full domain names by appending each TLD to each unique name
full_domains = []
for name in unique_names:
    for tld in tlds:
        full_domains.append(f"{name}.{tld}")

# Write the full domain names to a text file
with open("domains.txt", "w", encoding="utf-8") as f:
    for domain in full_domains:
        f.write(domain + "\n")
