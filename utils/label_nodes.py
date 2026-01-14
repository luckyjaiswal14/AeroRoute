import pandas as pd

# Load nodes from OSMN X output
df = pd.read_csv("data/airport_nodes_delhi.csv")

# Add a label column with default
df['label'] = 'Intersection'

# Label based on latitude (Y) – tune these values as needed
df.loc[df['y'] > 28.562, 'label'] = 'Gate'
df.loc[df['y'] < 28.55, 'label'] = 'Runway'

# Enumerate readable names
df['readable_label'] = df.groupby('label').cumcount()
df['readable_label'] = df['label'] + "_" + df['readable_label'].astype(str)

# Save for lookup later
df[['osmid', 'readable_label']].to_csv("data/node_labels.csv", index=False)
df.to_csv("data/labeled_airport_nodes_delhi.csv", index=False)
