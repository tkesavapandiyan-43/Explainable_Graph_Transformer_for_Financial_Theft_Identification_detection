import os

# Target phrase to search for
TARGET_PHRASE = "Dataset A - Meituan-style (generated): 4,000 accounts · 96,262 events · 422 fraud / 2,581 legitimate labelled · split train/val/test = 1,802 / 601 / 600 · identical split for every model"

# File extensions commonly found in project source files
SUPPORTED_EXTENSIONS = ('.tex', '.html', '.md', '.py', '.txt', '.json', '.jsx', '.tsx', '.css', '.cls', '.sty')

def find_target_phrase_in_project(root_dir='.'):
    matching_files = {}

    for dirpath, _, filenames in os.walk(root_dir):
        # Skip hidden directories like .git or node_modules
        if '.git' in dirpath or 'node_modules' in dirpath or '__pycache__' in dirpath:
            continue

        for filename in filenames:
            if filename.endswith(SUPPORTED_EXTENSIONS):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        matches = []
                        for line_num, line in enumerate(lines, 1):
                            if TARGET_PHRASE in line:
                                matches.append((line_num, line.strip()))
                        
                        if matches:
                            matching_files[file_path] = matches
                except Exception as e:
                    print(f"Could not read file {file_path}: {e}")

    return matching_files

if __name__ == '__main__':
    print("Searching project files...\n")
    results = find_target_phrase_in_project('.')

    if not results:
        print("No files found containing the specified text.")
    else:
        print(f"Found matches in {len(results)} file(s):\n" + "="*50)
        for filepath, matches in results.items():
            print(f"\n📁 File: {filepath}")
            for line_num, line_content in matches:
                print(f"   └── Line {line_num}: {line_content}")