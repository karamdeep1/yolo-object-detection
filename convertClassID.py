from pathlib import Path

labels_dir = Path("datasets/Teent.v3i.yolo26/valid/labels")

def convert_labels(root):
    for txt_file in root.rglob("*.txt"):
        try:
            txt_path = Path("\\\\?\\" + str(txt_file.resolve()))

            new_lines = []

            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    
                    if not parts:
                        continue

                    if parts[0] == "0":
                        parts[0] = "1"

                    new_lines.append(" ".join(parts))

            with open(txt_path, "w", encoding="utf-8") as f:
                for line in new_lines:
                    f.write(line + "\n")

            print(f"Updated: {txt_file}")

        except Exception as e:
            print(f"Error with file {txt_file}: {e}")

def main():
    convert_labels(labels_dir)
    print("Done converting labels!")

if __name__ == "__main__":
    main()