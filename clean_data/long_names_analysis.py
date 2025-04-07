import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('commonsld.csv')

long_names_df = df[df['sld'].str.len() > 9].copy()

# Function to analyze each sld string
def analyze_sld(sld):
    vowels_set = set("aeiou")
    vowel_count = sum(1 for char in sld.lower() if char in vowels_set)
    consonant_count = len(sld) - vowel_count
    return vowel_count, consonant_count

# Apply analysis on the long names
long_names_df['length'] = long_names_df['sld'].apply(len)
long_names_df['vowels'], long_names_df['consonants'] = zip(*long_names_df['sld'].apply(analyze_sld))
long_names_df['vowel_percentage'] = (long_names_df['vowels'] / long_names_df['length'] * 100).round(2)

# Graph 1: Bar chart for Occurrence of Long SLDs
plt.figure(figsize=(8, 6))
plt.bar(long_names_df['sld'], long_names_df['occurrence_count'])
plt.xlabel('SLD')
plt.ylabel('Occurrence')
plt.title('Occurrences for SLDs Longer Than 9 Characters')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plot1.png")
plt.show()

# Graph 2: Bar chart comparing Length, Vowels, and Consonants
x = range(len(long_names_df))
width = 0.25

plt.figure(figsize=(8, 6))
plt.bar([p - width for p in x], long_names_df['length'], width, label='Length')
plt.bar(x, long_names_df['vowels'], width, label='Vowels')
plt.bar([p + width for p in x], long_names_df['consonants'], width, label='Consonants')
plt.xlabel('SLD')
plt.ylabel('Count')
plt.title('Length, Vowels, and Consonants in SLDs')
plt.xticks(x, long_names_df['sld'], rotation=45)
plt.legend()
plt.tight_layout()

plt.savefig("plot2.png")
plt.show()

# Graph 3: Bar chart for Vowel Percentage
plt.figure(figsize=(8, 6))
plt.bar(long_names_df['sld'], long_names_df['vowel_percentage'])
plt.xlabel('SLD')
plt.ylabel('Vowel Percentage (%)')
plt.title('Vowel Percentage in SLDs Longer Than 9 Characters')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("plo3.png")
plt.show()