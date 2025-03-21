#resolved analysis: analysis on names that are resolved (all and newly discovered)

import csv
from collections import Counter
import matplotlib.pyplot as plt

# CSV file with full domain names
csv_file = "letter_resolved_domains.csv"
# Set to hold unique SLD labels
unique_slds = set()

with open(csv_file, "r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader, None)  # Skip header row if present
    for row in reader:
        # Normalize the full domain name
        full_domain = row[0].strip().lower()
        # Extract the SLD: take the part before the first dot
        sld = full_domain.split('.')[0]
        unique_slds.add(sld)

# Analyze SLD labels
lengths = [len(sld) for sld in unique_slds]
average_length = sum(lengths) / len(lengths) if lengths else 0
min_length = min(lengths) if lengths else 0
max_length = max(lengths) if lengths else 0

print("\nSLD Length Statistics:")
print(f"Average length: {average_length:.2f}")
print(f"Minimum length: {min_length}")
print(f"Maximum length: {max_length}")

# Compute overall character distribution across all unique SLDs
char_distribution = Counter()
for sld in unique_slds:
    char_distribution.update(sld)

print("\nCharacter Distribution:")
for char in sorted(char_distribution):
    print(f"{char}: {char_distribution[char]}")

# Plotting the SLD length distribution and character frequency distribution
plt.figure(figsize=(14, 6))

# Subplot 1: Histogram for SLD lengths
plt.subplot(1, 2, 1)
plt.hist(lengths, bins=range(min_length, max_length + 2), edgecolor='black', alpha=0.7)
plt.title('Distribution of SLD Lengths')
plt.xlabel('SLD Length')
plt.ylabel('Frequency')

# Subplot 2: Bar chart for character distribution
plt.subplot(1, 2, 2)
# Sort characters alphabetically for plotting
chars = sorted(char_distribution.keys())
frequencies = [char_distribution[char] for char in chars]
plt.bar(chars, frequencies, color='skyblue', edgecolor='black')
plt.title('Character Frequency Distribution in SLDs')
plt.xlabel('Character')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig("plot.png")
plt.show()
