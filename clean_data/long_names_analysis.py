import pandas as pd
import matplotlib.pyplot as plt
import wordninja

df = pd.read_csv('commonsld.csv')

long_names_df = df[df['sld'].str.len() > 12].copy()

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

long_names = long_names_df.copy()
long_names['words'] = long_names['sld'].apply(split_and_filter)
long_names['word_count'] = long_names['words'].apply(len)
long_names['length'] = long_names['sld'].apply(len)
long_names['vowels'], long_names['consonants'] = zip(*long_names['sld'].apply(analyze_sld))
long_names['vowel_percentage'] = (long_names['vowels'] / long_names['length'] * 100).round(2)

print("Long SLDs with word segmentation (only words >= 3 letters):")
print(long_names[['sld', 'words', 'word_count']])

print("Analysis of SLDs with more than 9 characters:")
print(long_names[['sld', 'length', 'vowels', 'consonants', 'vowel_percentage', 'occurrence_count']])

long_names.to_csv('long_name_analysis.csv', index=False)

long_names['word_count'].hist(bins=20)
plt.savefig('word_count_histogram.png')
plt.show()