import pandas as pd
import matplotlib.pyplot as plt
import wordninja

df = pd.read_csv('popularsld.csv')

long_names_df = df[df['sld'].str.len() <= 12].copy()

# Function to analyze each sld string
def analyze_sld(sld):
    vowels_set = set("aeiou")
    vowel_count = sum(1 for char in sld.lower() if char in vowels_set)
    consonant_count = len(sld) - vowel_count
    return vowel_count, consonant_count

def split_and_filter(sld):
    words = wordninja.split(sld)
    words = [w for w in words if len(w) >= 3]
    return words

def vowel_count():
    long_names = long_names_df.copy()
    long_names['words'] = long_names['sld'].apply(split_and_filter)
    long_names['word_count'] = long_names['words'].apply(len)
    long_names['length'] = long_names['sld'].apply(len)
    long_names['vowels'], long_names['consonants'] = zip(*long_names['sld'].apply(analyze_sld))
    long_names['vowel_percentage'] = (long_names['vowels'] / long_names['length'] * 100).round(2)

    print("Long SLDs with word segmentation (only words >= 3 letters):")
    print(long_names[['sld', 'words', 'word_count']])

    print("Analysis of SLDs with more than 9 characters:")
    print(long_names[['sld', 'length', 'vowels', 'consonants', 'vowel_percentage', 'occurrence']])

    long_names.to_csv('short_name_analysis.csv', index=False)

    long_names['word_count'].hist()
    plt.savefig('short_word_count_histogram.png')
    plt.show()

import pandas as pd

# Load the CSV file generated previously (e.g., "slds_with_readability.csv").
# Adjust the filename if necessary.
csv_file = "slds_with_readability.csv"
df = pd.read_csv(csv_file, dtype={"sld": str})

# Filter SLDs to only those longer than 12 characters.
# (We also check that the "sld" column is not null.)
df_filtered = df[df["sld"].notna() & (df["sld"].str.len() > 2)]

# If your readability score was only computed for long names, you may want
# to further filter out rows with no computed readability (NaN).
df_filtered = df_filtered[df_filtered["readability"].notna()]

# Calculate summary statistics for the readability scores.
if not df_filtered.empty:
    stats = df_filtered["readability"].describe()
    print("Summary statistics for readability scores (SLDs longer than 12 characters):")
    print(stats)
else:
    print("No SLDs longer than 12 characters with computed readability found.")

# Optionally, save the filtered DataFrame to a new CSV for further use.
output_csv = "filtered_slds.csv"
df_filtered.to_csv(output_csv, index=False)
print(f"\nFiltered results saved to '{output_csv}'")
