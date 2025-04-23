import json
import re
import nltk
from collections import Counter

# Uncomment the following line if you haven't already downloaded the Brown corpus:
# nltk.download('brown')

from nltk.corpus import brown

# Extract all words from the Brown corpus and convert to a single lowercase string.
text = " ".join(brown.words()).lower()

# Remove any characters that are not lowercase letters.
clean_text = re.sub(r"[^a-z]+", "", text)

# Generate a list of all consecutive two-letter combinations (bigrams).
bigram_list = [clean_text[i:i+2] for i in range(len(clean_text)-1)]

# Count the frequency of each bigram.
bigram_counts = Counter(bigram_list)

# Optionally, you can inspect a few sample key-value pairs:
for k, v in list(bigram_counts.items())[:5]:
    print(f"{k}: {v}")

# Save the bigram frequency dictionary to a JSON file.
with open("en_bigrams.json", "w") as f:
    json.dump(bigram_counts, f)

print("Bigram frequency JSON file 'en_bigrams.json' created successfully.")

import nltk
from nltk.corpus import words

# Download the words corpus if not already available.
nltk.download("words")

# Get the set of English words
word_list = words.words()

# Optionally, sort or filter the words.
# For example, you might want to sort them alphabetically:
word_list_sorted = sorted(word_list)

# Write the words to "words.txt", one per line.
with open("words.txt", "w") as f:
    for word in word_list_sorted:
        f.write(word.lower() + "\n")  # using lower-case to match our SLD analysis

print("words.txt created successfully.")
