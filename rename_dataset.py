from pathlib import Path

base = Path("./data/")

for i, jpg in enumerate(base.rglob("*.jpg"), start=1):
    new_name = jpg.with_name(f"image_{i}.jpg")
    jpg.rename(new_name)