import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Example dataset
dataset = [
    ['milk', 'bread', 'jam'],
    ['milk', 'bread'],
    ['milk', 'jam'],
    ['bread', 'jam'],
    ['milk', 'bread', 'butter'],
    ['bread', 'butter']
]

# Convert dataset into a DataFrame with one-hot encoding
import warnings

warnings.filterwarnings("ignore")
def encode_transactions(dataset):
    """ Convert list of transactions into a one-hot encoded DataFrame. """
    # Create a set of all unique items
    items = sorted(set(item for transaction in dataset for item in transaction))
    encoded_df = pd.DataFrame(0, index=range(len(dataset)), columns=items)
    for idx, transaction in enumerate(dataset):
        for item in transaction:
            encoded_df.at[idx, item] = 1
    return encoded_df

# One-hot encode the dataset
encoded_df = encode_transactions(dataset)
print("One-hot encoded dataset:")
print(encoded_df)

# Apply the Apriori algorithm to find frequent itemsets
min_support = 0.3  # Minimum support threshold
frequent_itemsets = apriori(encoded_df, min_support=min_support, use_colnames=True)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Generate association rules from the frequent itemsets
min_confidence = 0.7  # Minimum confidence threshold
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

print("\nAssociation Rules:")
print(rules)

# For clearer output of the rules
def print_rules(rules):
    for index, rule in rules.iterrows():
        antecedents = ", ".join(list(rule['antecedents']))
        consequents = ", ".join(list(rule['consequents']))
        print(f"Rule: {antecedents} -> {consequents} (support: {rule['support']:.2f}, confidence: {rule['confidence']:.2f})")

print("\nGenerated Association Rules:")
print_rules(rules)
