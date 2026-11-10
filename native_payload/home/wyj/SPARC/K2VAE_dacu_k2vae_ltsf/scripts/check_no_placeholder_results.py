import argparse
from pathlib import Path


PLACEHOLDERS = ["TBD", "PLACEHOLDER", "TODO_RESULT"]


def main():
    parser = argparse.ArgumentParser(description="Fail if paper still contains result placeholders.")
    parser.add_argument("paper_dir", type=Path)
    args = parser.parse_args()

    offenders = []
    candidates = [args.paper_dir / "iclr2026_conference.tex"]
    candidates.extend((args.paper_dir / "sections").glob("*.tex"))
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for token in PLACEHOLDERS:
            if token in text:
                offenders.append((path, token))

    if offenders:
        for path, token in offenders:
            print(f"{path}: contains {token}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
