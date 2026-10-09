"""Print all saved stages for one pipeline GUID, ordered by timestamp."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from uuid import UUID


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("guid", type=UUID)
    parser.add_argument("--log-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "logs" / "pipelines")
    args = parser.parse_args()
    directory = args.log_dir / str(args.guid)
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        print(f"Pipeline not found: {args.guid}", file=sys.stderr)
        return 1
    records = []
    try:
        for path in files:
            records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Cannot read pipeline log: {type(exc).__name__}", file=sys.stderr)
        return 2
    records.sort(key=lambda item: item["timestamp"])
    print(json.dumps(records, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
