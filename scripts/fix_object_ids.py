import pandas as pd
from pathlib import Path

def fix_dataset(manifest_path, output_dir):
    df = pd.read_csv(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate ID in the format expected by the model (e.g. 03008_c)
    def get_id(row):
        base = str(row['base_object_id'])
        variant = str(row['variant_id'])
        suffix = variant[-1] # 'c' or '0' -> '0' becomes 'b' in some logic?
        # Let's check the inference logic again.
        # Line 78 in inference.py: base_id, suffix = object_id.rsplit("_", 1)
        # Line 79: mesh_name = "model_c.ply" if suffix == "c" else "model_b_0.ply"
        # So it expects _c or _b.
        return "{}_{}".format(base, 'c' if suffix == 'c' else 'b')

    ids = df.apply(get_id, axis=1).tolist()
    
    output_path = output_dir / "object_ids.txt"
    output_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    print("Created {} with {} IDs".format(output_path, len(ids)))

if __name__ == "__main__":
    fix_dataset("data/manifests/fantastic_breaks_classification.csv", "data/fb_3d")
    fix_dataset("data/manifests/breaking_bad_classification.csv", "data/bb_3d")
