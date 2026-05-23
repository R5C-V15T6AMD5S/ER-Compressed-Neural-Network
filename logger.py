"""
logger.py — Shared logging utility for all 4 people in the project.

Creates a logs/ folder in the project root and saves a timestamped .txt
file every time a script is run. Each person gets their own log file.

Log file naming:
    logs/person1_20260522_1420.txt
    logs/person2_20260522_1435.txt
    logs/person3_20260522_1450.txt
    logs/person4_20260522_1505.txt
    logs/main_20260522_1510.txt    ← when running the full pipeline

Usage (add to the top of any script):
    from logger import Logger
    log = Logger("person1")       # or "person2", "person3", "person4", "main"

    log("Training started")
    log(f"Epoch 1 — Loss: 0.4321 | Accuracy: 67.50%")
    log.section("RESULTS")
    log(f"Final accuracy: 82.10%")
    log.close()                   # writes the file — call at the very end

The log is also printed to the terminal at the same time, so nothing changes
in how you see output — it just also gets saved.
"""

import os
import sys
from datetime import datetime


class Logger:
    """
    Captures all logged output and saves it to logs/<person>_<timestamp>.txt

    Parameters
    ----------
    person : str
        Label for the log file, e.g. "person1", "person2", "main"
    logs_dir : str
        Folder to save logs in (default: ./logs relative to working directory)
    also_print : bool
        If True (default), also prints each line to the terminal
    """

    def __init__(self, person="person1", logs_dir="./logs", also_print=True):
        self.person      = person
        self.also_print  = also_print
        self.lines       = []

        # Create logs/ folder if it doesn't exist
        os.makedirs(logs_dir, exist_ok=True)

        # Timestamped filename: person1_20260522_1420.txt
        timestamp     = datetime.now().strftime("%Y%m%d_%H%M")
        filename      = f"{person}_{timestamp}.txt"
        self.filepath = os.path.join(logs_dir, filename)

        # Write header
        header_lines = [
            "=" * 60,
            f"  Project: CNN Model Compression",
            f"  Person : {person}",
            f"  Date   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Log    : {self.filepath}",
            "=" * 60,
            "",
        ]
        for line in header_lines:
            self._record(line)

    def __call__(self, text=""):
        """log(text) — log a line of text."""
        self._record(str(text))

    def section(self, title):
        """log.section('RESULTS') — prints a bold section divider."""
        self._record("")
        self._record("─" * 60)
        self._record(f"  {title}")
        self._record("─" * 60)

    def table(self, rows, col_width=16):
        """
        log.table([
            ["Metric",   "Original", "Pruned"],
            ["Accuracy", "82.10%",   "79.40%"],
        ])
        Logs a simple text table.
        """
        for row in rows:
            line = "  " + "".join(str(cell).ljust(col_width) for cell in row)
            self._record(line)

    def close(self):
        """Write everything to the log file. Call at the very end of your script."""
        self._record("")
        self._record("=" * 60)
        self._record(f"  Log complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._record("=" * 60)

        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))

        # Print where the log was saved (to terminal only, not re-logged)
        print(f"\n  ✓ Log saved → {self.filepath}")

    def _record(self, text):
        self.lines.append(text)
        if self.also_print:
            print(text)


# ── Convenience: redirect print() to the logger ──────────────────────────────

class PrintCapture:
    """
    Optional: redirect all print() calls to the logger automatically.

    Usage:
        log = Logger("person1")
        sys.stdout = PrintCapture(log)
        # ... rest of your script — all print() calls now also go to the log
        log.close()
        sys.stdout = sys.__stdout__
    """
    def __init__(self, logger):
        self._logger  = logger
        self._stdout  = sys.__stdout__

    def write(self, text):
        text = text.rstrip("\n")
        if text:
            self._logger.lines.append(text)
        self._stdout.write(text + "\n" if text else "")

    def flush(self):
        self._stdout.flush()


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log = Logger("person1", logs_dir="./logs")

    log.section("TRAINING")
    log("Epoch  1/10 — Loss: 2.9801 | Accuracy: 12.40%")
    log("Epoch  2/10 — Loss: 2.1034 | Accuracy: 34.70%")
    log("Epoch  3/10 — Loss: 1.5621 | Accuracy: 55.20%")

    log.section("RESULTS")
    log.table([
        ["Metric",       "Original",  "Pruned",    "Quantized"],
        ["Accuracy (%)", "82.10",     "79.40",     "81.30"    ],
        ["Size (KB)",    "5012.0",    "2506.0",    "1253.0"   ],
        ["Inference ms", "38.2",      "36.1",      "37.0"     ],
        ["Sparsity (%)", "0.0",       "50.0",      "0.0"      ],
    ])

    log.close()
    print(f"\nOpen the file to check: logs/")
