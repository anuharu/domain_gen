import math
import re
import json
import pathlib
import pandas as pd
import pronouncing  # pip install pronouncing
import wordninja   # pip install wordninja

# Define vowels for vowel ratio calculation
VOWELS = set("aeiou")

def bigram_score(s, bigram_freq):
    """
    Calculate a bonus score for the SLD based on the likelihood of its bigrams.
    The score is mapped to a 0–20 range using logarithmic scaling.
    """
    if len(s) < 2:
        return 0
    # Sum the frequency counts for each bigram in the string; default to 1 if not found.
    score = sum(bigram_freq.get(s[i:i+2], 1) for i in range(len(s)-1))
    # Map the score to a value between 0 and 20 using logarithmic scaling (if score > 0)
    return min(20, 2 * math.log10(score)) if score > 0 else 0

def readability_sld(sld: str, lexicon, bigram_freq) -> float:
    """
    Calculates a readability score (0–100) for a given second-level domain (SLD)
    based on several factors:
    
      - Length penalty: optimal length is around 8 characters.
      - Character penalty: subtract if SLD includes any non-letter characters.
      - Vowel ratio: penalizes SLDs whose ratio of vowels deviates from ~0.4.
      - Dictionary bonus: reward if the SLD itself is in the lexicon.
      - Pronounceability bonus: reward if the CMU Pronouncing Dictionary finds a pronunciation.
      - Word segmentation bonus: reward if wordninja can break the SLD into multiple valid dictionary words.
      - Bigram likelihood: reward based on the frequency of its two-letter sequences.
    """
    s = sld.lower()
    base = 100.0

    # 1. Length penalty (ideal length ≈ 8 characters)
    base -= 2 * abs(len(s) - 8)

    # 2. Character-set penalty: subtract 15 if any character is not a-z
    if re.search(r"[^a-z]", s):
        base -= 15

    # 3. Vowel ratio penalty: ideal ratio is approximately 0.4
    v_ratio = sum(c in VOWELS for c in s) / max(1, len(s))
    base -= 50 * abs(v_ratio - 0.4)

    # 4. Dictionary bonus: reward if the entire SLD is a valid word
    if s in lexicon:
        base += 15

    # 5. Pronounceability bonus: using CMUdict via 'pronouncing'
    if pronouncing.phones_for_word(s):
        base += 10

    # 6. Word segmentation bonus: use wordninja to split the SLD into component words
    words_seg = wordninja.split(s)
    if len(words_seg) > 1 and all(word in lexicon for word in words_seg):
        # Award an extra bonus if the SLD nicely splits into known dictionary words.
        base += 10

    # 7. Bigram likelihood bonus: reward based on common bigrams in English
    base += bigram_score(s, bigram_freq)

    # Clip the final score to the 0–100 range
    return max(0, min(100, base))

def load_resources(bigram_file="en_bigrams.json", lexicon_file="words.txt"):
    """
    Loads required resources:
      - Bigram frequencies from a JSON file.
      - A lexicon from a text file (one word per line).
    """
    # Load bigram frequency counts
    with open(bigram_file, "r") as f:
        bigram_freq = json.load(f)
    # Load lexicon words (converted to lowercase)
    with open(lexicon_file, "r") as f:
        lexicon = {line.strip().lower() for line in f if line.strip()}
    return lexicon, bigram_freq

def analyze_slds_from_csv(csv_file, output_csv="slds_with_readability.csv"):
    lexicon, bigram_freq = load_resources()
    # Read CSV with dtype=str to force SLDs to be strings.
    df = pd.read_csv(csv_file, dtype={"sld": str})
    
    # Alternatively, if you don't want to change the CSV read, you can:
    # df["readability"] = df["sld"].apply(
    #     lambda s: readability_sld(str(s), lexicon, bigram_freq) if pd.notnull(s) else None
    # )

    # Compute readability score for each SLD
    df["readability"] = df["sld"].apply(
        lambda s: readability_sld(s, lexicon, bigram_freq) if isinstance(s, str) else None
    )

    stats = df["readability"].describe()
    print("Readability Summary Statistics:")
    print(stats)

    df_filtered = df[df["readability"] >= 40]
    print("\nNumber of SLDs with readability >= 40:", len(df_filtered))
    
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to '{output_csv}'")
    return df

# ---- Example usage: ----
# This portion can be run as a script, assuming you have a CSV file (e.g., "slds.csv")
# with a column called "sld", and the resource files "en_bigrams.json" and "words.txt" are present.
if __name__ == "__main__":
    import sys
    # Allow CSV file input from command-line; default to "slds.csv" if none provided
    csv_input = sys.argv[1] if len(sys.argv) > 1 else "slds.csv"
    analyze_slds_from_csv(csv_input)
