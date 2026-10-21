"""Verify the assembled release against its SHA-256 manifest (stdlib only)."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = ROOT / "MANIFEST.sha256"
    failures = []
    count = 0
    for line in manifest.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        count += 1
        if digest.hexdigest() != expected:
            failures.append(f"modified: {relative}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {count} files against MANIFEST.sha256.")


if __name__ == "__main__":
    main()
