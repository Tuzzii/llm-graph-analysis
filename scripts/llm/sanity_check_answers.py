import glob

files = glob.glob("results/answers/**/*jsonl", recursive=True)
print("Total answer files:", len(files))
for f in files:
    print("→", f)
