import csv
import pandas as pd

file_path = "data/raw/SparkText_SampleDataset_19681Abstracts.csv"

labels = []
texts = []

with open(file_path, "r", encoding="latin1") as f:
    reader = csv.reader(f, delimiter=" ", quotechar='"')
    next(reader)  # skip header row

    for row in reader:
        if len(row) < 2:
            continue  # skip any malformed/empty rows
        label = row[0].strip()
        text = row[1].strip()
        if label and text:
            labels.append(label)
            texts.append(text)

df = pd.DataFrame({"label": labels, "text": texts})

print("Total rows loaded:", len(df))
print("\nLabel distribution:")
print(df["label"].value_counts())
print("\nSample row:")
print(df.iloc[0])

# Save a clean version so we never have to re-parse the messy raw file again
df.to_csv("data/processed/clean_abstracts.csv", index=False)
print("\nSaved cleaned file to data/processed/clean_abstracts.csv")