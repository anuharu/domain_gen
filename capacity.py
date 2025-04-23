import argparse
import pandas as pd
import dns.resolver
import concurrent.futures
import sys

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate how many generated SLDs are novel and actually resolve"
    )
    p.add_argument("--generated-file", required=True,
                   help="Path to TXT file with one generated SLD per line")
    p.add_argument("--known-csv", required=True,
                   help="Path to CSV of known resolved SLDs")
    p.add_argument("--known-column", default="sld",
                   help="Column name in known CSV containing the SLDs (default: sld)")
    p.add_argument("--tld", default="com",
                   help="TLD to append for resolution checks (default: com)")
    p.add_argument("--max-workers", type=int, default=20,
                   help="Number of threads for DNS lookups (default: 20)")
    p.add_argument("--save", action="store_true",
                   help="Save seen/novel/novel_resolving lists to disk")
    p.add_argument("--output-prefix", default="results",
                   help="Prefix for output files if --save is set (default: results)")
    return p.parse_args()

def load_generated(path):
    with open(path, "r", encoding="utf-8") as f:
        # strip whitespace, drop empties, dedupe
        names = {line.strip() for line in f if line.strip()}
    return names

def load_known(path, column):
    df = pd.read_csv(path, usecols=[column])
    return set(df[column].astype(str).str.strip().tolist())

def resolves(sld, tld="com"):
    """Return True if sld.<tld> has at least one A record."""
    try:
        dns.resolver.resolve(f"{sld}.{tld}", "A")
        return True
    except Exception:
        return False

def main():
    args = parse_args()

    print("Loading generated names…", file=sys.stderr)
    generated = load_generated(args.generated_file)
    print(f"  {len(generated)} unique generated SLDs", file=sys.stderr)

    print("Loading known resolved SLDs…", file=sys.stderr)
    known = load_known(args.known_csv, args.known_column)
    print(f"  {len(known)} known SLDs", file=sys.stderr)

    seen  = {s for s in generated if s in known}
    novel = generated - seen
    print(f"\nSample size:          {len(generated)}")
    print(f"– seen in known set:  {len(seen)} ({len(seen)/len(generated):.1%})")
    print(f"– novel:              {len(novel)} ({len(novel)/len(generated):.1%})\n")

    print(f"Checking DNS resolution for {len(novel)} novel names (.{args.tld})…", file=sys.stderr)
    novel_resolving = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # map each novel name to a future
        futures = {executor.submit(resolves, sld, args.tld): sld for sld in novel}
        for fut in concurrent.futures.as_completed(futures):
            sld = futures[fut]
            if fut.result():
                novel_resolving.append(sld)

    print(f"– novel & resolving:  {len(novel_resolving)} ({len(novel_resolving)/len(generated):.1%})")
    if novel:
        print(f"– resolution rate among novel: {len(novel_resolving)/len(novel):.1%}")

    if args.save:
        prefix = args.output_prefix
        print(f"\nSaving lists to:\n  {prefix}_seen.txt\n  {prefix}_novel.txt\n  {prefix}_novel_resolving.txt")
        with open(f"{prefix}_seen.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(seen)))
        with open(f"{prefix}_novel.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(novel)))
        with open(f"{prefix}_novel_resolving.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(novel_resolving)))

if __name__ == "__main__":
    main()