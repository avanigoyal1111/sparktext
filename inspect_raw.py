import csv

file_path = "data/raw/SparkText_SampleDataset_19681Abstracts.csv"

with open(file_path, "r", encoding="latin1") as f:
    reader = csv.reader(f, delimiter=" ", quotechar='"')
    header = next(reader)
    print("Header row (raw):", header)

    for i, row in enumerate(reader):
        print(f"--- Row {i} ---")
        print("Number of fields:", len(row))
        print("Field 0 (label):", row[0])
        print("Field 1 preview (first 150 chars):", row[1][:150] if len(row) > 1 else "MISSING")
        print()
        if i >= 3:
            break