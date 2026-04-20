# Author: CHATGPT,
# CREATED BY: RAPHAEL XIE

import argparse
import json
import re
from collections import Counter
from datetime import datetime

DT_FMT = "%Y-%m-%d %H:%M:%S"

def parse_dt(s: str):
    return datetime.strptime(s, DT_FMT)

def load_names(path, self_name):
    names = []
    with open(path) as f:
        for line in f:
            n = line.strip()
            if n and n != self_name:
                names.append(n)
    # 为了避免子串误匹配，按长度从长到短匹配
    names.sort(key=len, reverse=True)
    return names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True, help="path to associative_memory/nodes.json")
    ap.add_argument("--names", required=True, help="path to persona_names.txt (one per line)")
    ap.add_argument("--self", dest="self_name", required=True, help="e.g., 'Isabella Rodriguez'")
    ap.add_argument("--until", default=None, help="cutoff datetime, format 'YYYY-MM-DD HH:MM:SS'")
    ap.add_argument("--require-self-mention", action="store_true",
                    help="only count events that mention self_name somewhere")
    args = ap.parse_args()

    cutoff = parse_dt(args.until) if args.until else None
    names = load_names(args.names, args.self_name)

    with open(args.nodes) as f:
        data = json.load(f)

    # nodes.json structure: { "node_912": {...}, "node_911": {...}, ... }
    events = []
    for _, node in data.items():
        created = node.get("created")
        if not created:
            continue
        try:
            t = parse_dt(created)
        except Exception:
            continue
        if cutoff and t > cutoff:
            continue

        # build a searchable text blob
        fields = [
            str(node.get("subject", "")),
            str(node.get("predicate", "")),
            str(node.get("object", "")),
            str(node.get("description", "")),
        ]
        blob = " | ".join(fields)

        if args.require_self_mention and (args.self_name not in blob):
            continue

        events.append((t, blob, node.get("type", "")))

    events.sort(key=lambda x: x[0])

    tie_counts = Counter()
    last_seen = {}

    for t, blob, _typ in events:
        for n in names:
            if n in blob:
                tie_counts[n] += 1
                last_seen[n] = t

    print(f"Self: {args.self_name}")
    if cutoff:
        print(f"Cutoff: {args.until}")
    print(f"Total events scanned: {len(events)}")
    print(f"Alters found: {len(tie_counts)}\n")

    for n, c in tie_counts.most_common():
        ls = last_seen[n].strftime(DT_FMT)
        print(f"- {n}: count={c}, last_seen={ls}")

if __name__ == "__main__":
    main()
