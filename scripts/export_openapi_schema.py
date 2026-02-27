from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manage_py = repo_root / "src" / "webapp" / "manage.py"
    output_path = repo_root / "openapi.json"

    subprocess.run(
        [sys.executable, str(manage_py), "spectacular", "--file", str(output_path)],
        check=True,
    )
    print(f"OpenAPI schema exported: {output_path}")


if __name__ == "__main__":
    main()
