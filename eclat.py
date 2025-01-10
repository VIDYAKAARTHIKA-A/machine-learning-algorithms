import pandas as pd
from itertools import combinations

# Define a sample dataset
data = {
    'Transaction': [1, 2, 3, 4, 5],
    'Items': [
        ['Milk', 'Bread', 'Butter'],
        ['chesse', 'Bread'],
        ['Milk', 'Bread', 'Butter', 'cheese'],
        ['Milk', 'Bread'],
        ['Butter', 'cheese']
    ]
}

# Convert the dataset to a DataFrame
df = pd.DataFrame(data)
print("Original Dataset:")
print(df)

# Convert the dataset into a format suitable for Eclat (itemset and transaction IDs)
transactions = df.set_index('Transaction')['Items'].to_dict()
print("\nTransaction Dictionary:")
print(transactions)

# Eclat algorithm implementation
def eclat(transactions, min_support):
    def find_subsets(itemset):
        """ Helper function to find all non-empty subsets of an itemset """
        return [combinations(itemset, i) for i in range(1, len(itemset)+1)]
    
    # Initialize variables
    itemsets = {}
    frequent_itemsets = []
    
    # Step 1: Generate initial itemsets with transaction IDs
    for tid, items in transactions.items():
        for item in items:
            if item in itemsets:
                itemsets[item].add(tid)
            else:
                itemsets[item] = {tid}

    # Step 2: Find frequent itemsets
    for itemset, tids in itemsets.items():
        support = len(tids)
        if support >= min_support:
            frequent_itemsets.append((frozenset([itemset]), support))
    
    # Step 3: Generate larger itemsets by combining frequent itemsets
    k = 2
    current_itemsets = {frozenset([item]): tids for item, tids in itemsets.items() if len(tids) >= min_support}
    while current_itemsets:
        new_itemsets = {}
        for itemset1 in current_itemsets:
            for itemset2 in current_itemsets:
                if itemset1 != itemset2:
                    new_itemset = itemset1 | itemset2
                    if len(new_itemset) == k:
                        tids = current_itemsets[itemset1] & current_itemsets[itemset2]
                        if len(tids) >= min_support:
                            new_itemsets[new_itemset] = tids
                            frequent_itemsets.append((new_itemset, len(tids)))
        current_itemsets = new_itemsets
        k += 1
    
    return frequent_itemsets

# Define the minimum support threshold
min_support = 2

# Run the Eclat algorithm
frequent_itemsets = eclat(transactions, min_support)
frequent_itemsets = sorted(frequent_itemsets, key=lambda x: -x[1])  # Sort by support in descending order

# Print the results
print("\nFrequent Itemsets and Their Supports:")
for itemset, support in frequent_itemsets:
    print(f"Itemset: {set(itemset)}, Support: {support}")


