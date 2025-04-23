#zdns_analysis.py: analyze the json output file (from running zdns on generated names), save all resolved names to CSV

import json
import csv

resolved_count = 0
total_count = 0
resolved = []

with open("new_output.json", "r", encoding="utf-8") as f:
    for line in f:
        total_count += 1
        data = json.loads(line)
        # "status" might be "NOERROR" if DNS resolution was successful
        a_data = data.get("results", {}).get("A", {})
        status = a_data.get("status")
        if status == "NOERROR":
            resolved_count += 1
            resolved.append(data.get("name"))
            
print(f"Domains queried: {total_count}")
print(f"Domains resolved: {resolved_count}")
print(f"Percentage that exist: {resolved_count / total_count * 100:.2f}%")

# Write all resolved names to a CSV file
with open("letter_resolved_domains.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["name"])  # header row
    for name in resolved:
        writer.writerow([name])
