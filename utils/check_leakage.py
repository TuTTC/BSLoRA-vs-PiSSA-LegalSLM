import os
import json
from datasets import load_dataset
import hashlib

def get_hash(premise, hypothesis):
    # Normalize and hash for comparison
    text = f"{premise.strip().lower()}|{hypothesis.strip().lower()}"
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def check_leakage():
    print("[INFO] Loading uitnlp/ViANLI dataset...")
    ds = load_dataset("uitnlp/ViANLI")
    
    train_set = ds["train"]
    test_set = ds["test"]
    
    print(f"[INFO] Train samples: {len(train_set)}")
    print(f"[INFO] Test samples: {len(test_set)}")
    
    # Store train hashes
    print("[INFO] Hashing train set...")
    train_hashes = {}
    for i, ex in enumerate(train_set):
        h = get_hash(ex["premise"], ex["hypothesis"])
        if h not in train_hashes:
            train_hashes[h] = []
        train_hashes[h].append(i)
        
    print("[INFO] Checking test set for overlaps...")
    overlaps = []
    for i, ex in enumerate(test_set):
        h = get_hash(ex["premise"], ex["hypothesis"])
        if h in train_hashes:
            overlaps.append({
                "test_idx": i,
                "train_indices": train_hashes[h],
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
                "label": ex["label"]
            })
            
    print(f"\n[RESULTS] Found {len(overlaps)} overlapping samples out of {len(test_set)} test samples.")
    print(f"[RESULTS] Leakage Percentage: {(len(overlaps)/len(test_set))*100:.2f}%")
    
    # Save detailed report
    report_path = "outputs/results/leakage_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "train_size": len(train_set),
                "test_size": len(test_set),
                "overlap_count": len(overlaps),
                "leakage_percent": (len(overlaps)/len(test_set))*100
            },
            "overlaps": overlaps
        }, f, indent=2, ensure_ascii=False)
        
    print(f"[INFO] Detailed report saved to: {report_path}")
    
    if len(overlaps) > 0:
        print("\n[WARNING] Data leakage detected! The 100% accuracy may be due to the model memorizing test samples.")
    else:
        print("\n[SUCCESS] No data leakage found between Train and Test sets.")

if __name__ == "__main__":
    check_leakage()
