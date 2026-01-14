import pandas as pd

# Load raw nodes
df = pd.read_csv("data/airport_nodes_delhi.csv")

# Predefined real labels
real_labels = {
    345277850: "Apron Stand A1",
    413760666: "Taxiway Alpha",
    413760670: "Taxiway Bravo",
    949748079: "Taxiway Charlie",
    1377318095: "Taxiway Delta",
    1794068380: "Terminal 3 Gate B1",
    1794068404: "Terminal 3 Gate B2",
    1794070257: "Runway Entry 28A",
    1794070301: "Runway Entry 28B",
    1987508656: "Runway Holding Point 28L",
    2353491223: "Runway Exit 28L-E1",
    2353491228: "Runway Exit 28L-E2",
    2353491237: "Runway Exit 28L-E3",
    4572503843: "Taxiway Echo",
    4572504342: "Taxiway Foxtrot",
    4572504360: "Runway Entry 11A",
    6883916918: "Terminal 1 Apron",
    10982104589: "Terminal 2 Apron",
    10982117547: "Cargo Terminal Apron",
    10982117566: "Maintenance Hangar",
    10982125359: "Refueling Station"
}

# Map labels
df["readable_label"] = df["osmid"].map(real_labels)

# Fallback for unknown nodes (optional)
df["readable_label"] = df["readable_label"].fillna("Unknown")

# Save output
df[["osmid", "readable_label"]].to_csv("data/node_labels.csv", index=False)
df.to_csv("data/labeled_airport_nodes_delhi.csv", index=False)

print("✅ node_labels.csv generated with real airport labels.")
