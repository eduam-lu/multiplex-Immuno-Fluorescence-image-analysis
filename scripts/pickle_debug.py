import torch
import os
import sys

# --- Usage ---
# python check_pt_files.py /path/to/folder
# -----------------

folder = sys.argv[1] if len(sys.argv) > 1 else "."
files = sorted([f for f in os.listdir(folder) if f.endswith(".pt")])

print(f"\n🔍 Checking {len(files)} .pt files in {folder}\n")

bad_files = []

for i, fname in enumerate(files, 1):
    fpath = os.path.join(folder, fname)
    print(f"[{i}/{len(files)}] {fname} ... ", end="", flush=True)
    try:
        torch.load(fpath, map_location="cpu")
        print("✅ OK")
    except Exception as e:
        print("❌ FAILED")
        bad_files.append((fname, str(e)))

if bad_files:
    print("\n⚠️  The following files could not be loaded:\n")
    for f, err in bad_files:
        print(f" - {f}: {err}")
else:
    print("\n✅ All files loaded successfully!")