"""Trim synthetic conversations to valid grid sizes."""

import os

def trim_conversation(input_file, output_file, target_count):
    """Trim a conversation file to exactly target_count messages."""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newline to get messages
    messages = [msg.strip() for msg in content.split('\n\n') if msg.strip()]

    # Trim to target count
    trimmed = messages[:target_count]

    # Write back
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(trimmed))

    print(f"{os.path.basename(output_file)}: {len(messages)} -> {len(trimmed)} messages")

def main():
    from pathlib import Path

    # Use relative path from script location
    data_dir = Path(__file__).parent / "data"

    # Target sizes: 16 (4×4), 20 (4×5), 12 (3×4)
    trim_specs = {
        "synthetic_code_review.txt": 16,  # Already 16
        "synthetic_cooking.txt": 20,      # 22 → 20
        "synthetic_science.txt": 20,      # 24 → 20
        "synthetic_travel.txt": 20,       # 26 → 20
        "synthetic_book_analysis.txt": 12  # 22 → 12
    }

    for filename, target in trim_specs.items():
        input_path = data_dir / filename
        output_path = input_path  # Overwrite in place
        trim_conversation(input_path, output_path, target)

    print("\nAll conversations trimmed to valid grid sizes!")
    print("Grid sizes: 4×4 (16), 4×5 (20), 4×5 (20), 4×5 (20), 3×4 (12)")

if __name__ == "__main__":
    main()
