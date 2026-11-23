# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import re
import sys
from datetime import datetime
from pathlib import Path

HEADER_TEMPLATE = "# Copyright (c) {year} Robotics and AI Institute LLC. All rights reserved."
# Also accepts a year range ("2025-2026", checked by its first year) and the "dba RAI Institute" trade name used by
# code vendored from other RAI repos (e.g. judo/unlocked from starfish).
HEADER_REGEX = r"^# Copyright \(c\) (\d{4})(?:-\d{4})? Robotics and AI Institute LLC(?: dba RAI Institute)?\. All rights reserved\."


def get_header_year(lines: list[str]) -> int | None:
    """Return the year from header if found in first 5 lines, else None."""
    for line in lines[:5]:
        m = re.match(HEADER_REGEX, line)
        if m:
            return int(m.group(1))
    return None


def add_header_to_file(file_path: str) -> int:
    """Add or update header in the Python file."""
    path = Path(file_path)
    if path.suffix != ".py":
        return 0

    content = path.read_text(encoding="utf-8").splitlines(keepends=True)
    current_year = datetime.now().year
    header_line_new = HEADER_TEMPLATE.format(year=current_year) + "\n"

    modified = False

    # respect existing shebang line
    insert_at = 1 if content and content[0].startswith("#!") else 0

    existing_year = get_header_year(content)
    if existing_year is not None:
        # header exists, check if year is wrong
        if existing_year not in range(2025, current_year + 1):
            # find the exact header line index and replace it
            for idx, line in enumerate(content[:5]):
                if re.match(HEADER_REGEX, line):
                    content[idx] = header_line_new
                    modified = True
                    break
        # else: year is up to date, no change needed
    else:
        # no header found, insert new header
        new_header = [header_line_new, "\n"]
        content = content[:insert_at] + new_header + content[insert_at:]
        modified = True

    if modified:
        path.write_text("".join(content), encoding="utf-8")
        return 1
    return 0


def main() -> int:
    """Main function to process files passed as arguments."""
    modified = 0
    for file_path in sys.argv[1:]:
        try:
            modified += add_header_to_file(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
