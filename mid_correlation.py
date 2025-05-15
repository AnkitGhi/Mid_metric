import json
from scipy.stats import kendalltau


def extract_scores(file_path):
    """
    Reads a tab-separated ratings file and returns a flat list of all scores.
    Each line is expected to have at least 5 columns:
      img1    img2#n    score1    score2    score3
    """
    scores = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()  # splits on any whitespace (tabs included)
            if len(parts) < 5:
                continue  # skip malformed lines
            # take the last three entries and convert to int
            scores.extend(int(x) for x in parts[-3:])
    return scores


file_path = "/Users/ankitghimire/Downloads/Flickr8k_text/ExpertAnnotations.txt"  # replace with your filename
all_scores = extract_scores(file_path)
print(len(all_scores))


# Replace with the path to your JSON file
json_path = '/Users/ankitghimire/Downloads/new_pairs_with_mid.json'



with open(json_path, 'r') as f:
    data = json.load(f)

# Extract all PMI values
pmi_values = [entry['mid_pmi_score'] for entry in data]
average_value = [entry['average_score'] for entry in data]


expanded = [x for x in pmi_values for _ in range(3)]
print(len(expanded))

# compute Kendall’s tau-c
tau_c, p_value = kendalltau(pmi_values, average_value, variant='c')

print(f"Kendall’s tau-c: {tau_c:.4f}")
print(f"two-sided p-value: {p_value:.4g}")






