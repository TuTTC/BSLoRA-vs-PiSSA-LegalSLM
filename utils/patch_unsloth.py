import os
import sys

def patch_file(file_path):
    if not os.path.exists(file_path):
        print(f"[ERROR] {file_path} not found.")
        return False

    with open(file_path, "r") as f:
        content = f.read()

    # The fix: Ensure RoPE cos/sin tensors are sliced to match the Query tensor length
    # This addresses the broadcast error: Qn *= cos [1, 32, 1, 128] vs [1, 32, N, 128]
    fix_pissa = (
        "                    if cos.shape[2] != Qn.shape[2]:\n"
        "                        cos = cos[:, :, -Qn.shape[2]:, :]\n"
        "                        sin = sin[:, :, -Qn.shape[2]:, :]\n"
        "                    Qn *= cos\n"
        "                    Kn *= cos"
    )

    # Search for the problematic line
    if "Qn *= cos" in content:
        if "if cos.shape[2] != Qn.shape[2]:" in content:
            print(f"[INFO] {file_path} already patched.")
            return True
        
        print(f"[PATCH] Patching {file_path}...")
        new_content = content.replace("Qn *= cos\n                    Kn *= cos", fix_pissa)
        
        # Also handle potential variations in indentation
        if new_content == content:
             new_content = content.replace("Qn *= cos\n                Kn *= cos", fix_pissa.replace("                    ", "                "))

        with open(file_path, "w") as f:
            f.write(new_content)
        print(f"[SUCCESS] {file_path} patched.")
        return True
    else:
        print(f"[WARNING] Could not find the Qn *= cos line in {file_path}. Maybe a different version?")
        return False

def main():
    # Attempt to find unsloth directory
    try:
        import unsloth
        unsloth_dir = os.path.dirname(unsloth.__file__)
    except ImportError:
        print("[ERROR] Unsloth not installed in current environment.")
        return

    print(f"[INFO] Detected unsloth at: {unsloth_dir}")
    
    # Files to patch
    files_to_patch = [
        os.path.join(unsloth_dir, "models", "qwen3.py"),
        os.path.join(unsloth_dir, "models", "llama.py")
    ]

    patched = False
    for f in files_to_patch:
        if patch_file(f):
            patched = True

    if not patched:
        print("[FAIL] No files were patched. Try updating unsloth via pip first.")
    else:
        print("[DONE] Environment fix applied. You can now run evaluation/evaluate.py")

if __name__ == "__main__":
    main()
