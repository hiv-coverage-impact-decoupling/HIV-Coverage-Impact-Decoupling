### ./extract_code.py
import os

def extract_all_code():
    print("[*] Code Extraction for EDI ")
    
    target_dirs = ["pipeline", "robustness", "tests", "utils", "exploratory"]
    root_files = ["run_pipeline.py", "extract_code.py"]
    output_file = "Master_Code_Log.txt"
    
    with open(output_file, 'w', encoding='utf-8') as out:
        for rf in root_files:
            if os.path.exists(rf):
                write_file(rf, out)
                
        for d in target_dirs:
            if os.path.exists(d):
                for f in sorted(os.listdir(d)):
                    if f.endswith(".py"):
                        filepath = os.path.join(d, f)
                        write_file(filepath, out)
                        
    print(f"[✓] Complete {output_file}")

def write_file(filepath, out):

    out.write("="*80 + "\n")
    out.write(f"FILE: ./{filepath.replace(chr(92), '/')}\n")
    out.write("="*80 + "\n\n")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            out.write(f.read() + "\n\n")
    except Exception as e:
        out.write(f"# [!] Cannot read file: {e}\n\n")

if __name__ == "__main__":
    extract_all_code()