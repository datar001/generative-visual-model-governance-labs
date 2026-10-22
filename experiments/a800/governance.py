"""Chapter 1: execute the existing governance exercise and preserve its actual CSV."""
from common import *
import subprocess, sys, csv, shutil


def main():
    with run("01_governance") as out:
        result = subprocess.run(
            [sys.executable, str(REPO / "chapters/01_overview/lab.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        (out / "stdout.txt").write_text(result.stdout, encoding="utf8")
        src = ROOT / "results/governance/risk_register.csv"
        shutil.copy2(src, out / "risk_register.csv")
        with src.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        write_json(
            out / "summary.json",
            dict(
                rows=len(rows),
                columns=list(rows[0]),
                records=rows,
                scope="Executed teaching scenarios, not a real product risk assessment.",
            ),
        )


if __name__ == "__main__":
    main()
