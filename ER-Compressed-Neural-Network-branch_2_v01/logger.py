"""
logger.py — Shared logging utility for all 4 people in the project.

Creates a logs/ folder in the project root and saves a timestamped .txt
file every time a script is run. Each person gets their own log file.

Also detects the OS at startup and prints a one-time info block so
everyone knows what system they are running on and any relevant notes
(e.g. Windows encoding fix, path separator style).

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
import platform
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
#  OS DETECTION — runs once when logger.py is imported
# ══════════════════════════════════════════════════════════════════════════════

def get_os_info():
    """
    Detect the current operating system and return a dict with:
        name        — 'Windows', 'Linux', or 'macOS'
        is_windows  — True/False
        is_linux    — True/False
        is_mac      — True/False
        version     — OS version string
        python      — Python version string
        sep         — path separator ('\\' on Windows, '/' on Linux/macOS)
        notes       — list of OS-specific notes to show the user
    """
    system = platform.system()   # 'Windows', 'Linux', 'Darwin'

    is_windows = system == "Windows"
    is_linux   = system == "Linux"
    is_mac     = system == "Darwin"

    name = {"Windows": "Windows", "Linux": "Linux", "Darwin": "macOS"}.get(system, system)

    notes = []

    if is_windows:
        notes.append("Windows detected — terminal may show garbled characters (encoding issue).")
        notes.append("The saved .txt log file will always be correct — open it in VS Code.")
        notes.append("Use backslashes or forward slashes in paths, both work in Python.")
        notes.append("To move an existing model file:  move cnn_fruits.npz models\\")

    elif is_linux:
        notes.append("Linux detected — full UTF-8 support, no encoding issues.")
        notes.append("To move an existing model file:  mv cnn_fruits.npz models/")

    elif is_mac:
        notes.append("macOS detected — full UTF-8 support, no encoding issues.")
        notes.append("To move an existing model file:  mv cnn_fruits.npz models/")

    return {
        "name":       name,
        "is_windows": is_windows,
        "is_linux":   is_linux,
        "is_mac":     is_mac,
        "version":    platform.version(),
        "python":     platform.python_version(),
        "sep":        os.sep,
        "notes":      notes,
    }


# Run detection immediately on import so every script gets it
OS = get_os_info()


def print_os_banner():
    """
    Print a one-time OS info block to the terminal.
    Called automatically by Logger.__init__ on first use.
    """
    print("┌" + "─" * 58 + "┐")
    print(f"│  System : {OS['name']:<47}│")
    print(f"│  Version: {platform.release():<47}│")
    print(f"│  Python : {OS['python']:<47}│")
    sep_note = f"'{OS['sep']}' (path separator on this OS)"
    print(f"│  Sep    : {sep_note:<47}│")
    print("├" + "─" * 58 + "┤")
    for note in OS["notes"]:
        # word-wrap at 56 chars
        words, line = note.split(), ""
        for word in words:
            if len(line) + len(word) + 1 > 56:
                print(f"│  {line:<56}│")
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            print(f"│  {line:<56}│")
    print("└" + "─" * 58 + "┘")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGER CLASS
# ══════════════════════════════════════════════════════════════════════════════

# Track whether the OS banner has been printed this session
_banner_printed = False


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
        global _banner_printed

        self.person     = person
        self.also_print = also_print
        self.lines      = []

        # Print OS banner once per session (terminal only, not logged)
        if not _banner_printed:
            print_os_banner()
            _banner_printed = True

        # Create logs/ folder if it doesn't exist
        os.makedirs(logs_dir, exist_ok=True)

        # Timestamped filename: person1_20260522_1420.txt
        timestamp     = datetime.now().strftime("%Y%m%d_%H%M")
        filename      = f"{person}_{timestamp}.txt"
        self.filepath = os.path.join(logs_dir, filename)

        # Header — includes OS info so the log file records what system was used
        header_lines = [
            "=" * 60,
            f"  Project: CNN Model Compression",
            f"  Person : {person}",
            f"  Date   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  OS     : {OS['name']} (Python {OS['python']})",
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
        """log.section('RESULTS') — prints a section divider."""
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
        Logs a simple aligned text table.
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

        # Always write as UTF-8 — fixes Windows terminal encoding issues
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))

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
        self._logger = logger
        self._stdout = sys.__stdout__

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
