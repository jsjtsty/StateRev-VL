from pathlib import Path
import json

root = Path("dataset/Theory-of-Space")

files = list(root.rglob("falsebelief_exp.json"))

print(f"Found {len(files)} falsebelief_exp.json files")

for p in files[:5]:
    print("\n" + "=" * 80)
    print("FILE:", p)

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("TYPE:", type(data))

    if isinstance(data, dict):
        print("KEYS:", list(data.keys())[:30])
        for k, v in list(data.items())[:2]:
            print("\nKEY:", k)
            print("VALUE:", repr(v)[:1500])

    elif isinstance(data, list):
        print("LENGTH:", len(data))
        for item in data[:2]:
            print("\nITEM:")
            print(repr(item)[:1500])