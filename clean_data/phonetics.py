import nltk
from nltk.corpus import cmudict
from collections import defaultdict, Counter
import math
import pandas as pd
import string

# -------------------------------
# Download and Load CMU Dictionary
# -------------------------------
nltk.download('cmudict')
cmu_dict = cmudict.dict()

# -------------------------------
# Build the Phonotactic Bigram Model
# -------------------------------
bigram_counts = defaultdict(Counter)
phoneme_counts = Counter()

for word in cmu_dict:
    for pron in cmu_dict[word]:
        pron = ['<s>'] + pron + ['</s>']
        for i in range(len(pron) - 1):
            bigram_counts[pron[i]][pron[i+1]] += 1
            phoneme_counts[pron[i]] += 1
        phoneme_counts[pron[-1]] += 1

vocab = list(phoneme_counts.keys())
V = len(vocab)

bigram_prob = defaultdict(dict)
for prev in bigram_counts:
    total = sum(bigram_counts[prev].values())
    for phon in vocab:
        bigram_prob[prev][phon] = (bigram_counts[prev][phon] + 1) / (total + V)

# -------------------------------
# Naive Grapheme-to-Phoneme (G2P) Converter
# -------------------------------
naive_letter_mapping = { 'a': ['AH'], 'b': ['B'], 'c': ['K'], 'd': ['D'], 'e': ['EH'],
                         'f': ['F'], 'g': ['G'], 'h': ['HH'], 'i': ['IH'], 'j': ['JH'],
                         'k': ['K'], 'l': ['L'], 'm': ['M'], 'n': ['N'], 'o': ['OW'],
                         'p': ['P'], 'q': ['K'], 'r': ['R'], 's': ['S'], 't': ['T'],
                         'u': ['UH'], 'v': ['V'], 'w': ['W'], 'x': ['K','S'], 'y': ['Y'], 'z': ['Z'] }
naive_digit_mapping = { '0': ['ZERO'], '1': ['ONE'], '2': ['TWO'], '3': ['THREE'],
                        '4': ['FOUR'], '5': ['FIVE'], '6': ['SIX'], '7': ['SEVEN'],
                        '8': ['EIGHT'], '9': ['NINE'] }

def naive_g2p(word):
    phonemes = []
    for char in word:
        lower_char = char.lower()
        if lower_char in naive_letter_mapping:
            phonemes.extend(naive_letter_mapping[lower_char])
        elif lower_char in naive_digit_mapping:
            phonemes.extend(naive_digit_mapping[lower_char])
        elif lower_char in string.whitespace:
            continue
        else:
            continue
    return phonemes

# -------------------------------
# Compute Phonotactic Probability
# -------------------------------
def phonotactic_probability(word):
    if not isinstance(word, str):
        return None
    word_lower = word.lower()
    if word_lower in cmu_dict:
        phoneme_sequence = cmu_dict[word_lower][0]
    else:
        phoneme_sequence = naive_g2p(word_lower)
        if not phoneme_sequence:
            return None
    pronunciation = ['<s>'] + phoneme_sequence + ['</s>']
    log_prob = 0.0
    for i in range(len(pronunciation) - 1):
        prev, curr = pronunciation[i], pronunciation[i+1]
        if prev in bigram_prob:
            prob = bigram_prob[prev].get(curr, 1 / (sum(bigram_counts[prev].values()) + V))
        else:
            prob = 1e-8
        log_prob += math.log(prob)
    return math.exp(log_prob)

# -------------------------------
# CSV Integration: Read and Process Words
# -------------------------------
def process_csv(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    if 'SLD' not in df.columns:
        raise ValueError("CSV file must contain a column named 'sld'.")
    df = df[df['SLD'].str.len() < 14]
    df = df[df['SLD'].apply(lambda x: isinstance(x, str))]
    results = []
    for word in df['SLD']:
        prob = phonotactic_probability(word)
        results.append({'word': word, 'phonotactic_probability': prob})
    result_df = pd.DataFrame(results)
    return result_df

# -------------------------------
# Main Execution Block
# -------------------------------
if __name__ == "__main__":
    csv_file = "../generated_names.csv"
    result_df = process_csv(csv_file)
    if result_df is not None:
        # Compute log probabilities
        result_df['log_phonotactic_probability'] = result_df['phonotactic_probability'].apply(
            lambda x: math.log(x) if x is not None and x > 0 else None)
        # Print full results
        print(result_df)
        # -------------------------------
        # Descriptive Statistics
        # -------------------------------
        stats = result_df['phonotactic_probability'].describe()
        stats_log = result_df['log_phonotactic_probability'].describe()
        print("\nDescriptive statistics for phonotactic_probability:\n", stats)
        print("\nDescriptive statistics for log_phonotactic_probability:\n", stats_log)
        # Optionally save stats
        stats.to_csv('phonotactic_stats.csv')
        stats_log.to_csv('phonotactic_log_stats.csv')
        # Save detailed results
        result_df.to_csv("1_phoneticresults.csv", index=False)