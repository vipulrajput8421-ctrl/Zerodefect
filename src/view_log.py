"""
view_log.py — CLI inspection log viewer.

Pulls metrics from SQLite and prints a summary breakdown of QC decisions
and recent inspection records.
"""

import sys
import argparse
from pathlib import Path

# Import utilities using relative import path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from logger import InspectionLogger
from utils import LOGS_DIR


def main():
    parser = argparse.ArgumentParser(description="ZeroDefect Inspection Log Viewer")
    parser.add_argument("--log-dir", type=str, default=str(LOGS_DIR), help="Folder with log files")
    parser.add_argument("--last", type=int, default=10, help="Number of recent logs to show")
    parser.add_argument("--export", type=str, default=None, help="Export a copy of the CSV to this path")
    parser.add_argument("--clear", action="store_true", help="Clear all database and CSV logs")
    args = parser.parse_args()

    logger = InspectionLogger(Path(args.log_dir))

    if args.clear:
        confirm = input("\n[WARNING] Are you sure you want to delete all log entries and thumbnail images? [y/N]: ")
        if confirm.lower().strip() == 'y':
            if logger.clear():
                print("[*] Log database and CSV cleared successfully.")
            else:
                print("[ERROR] Failed to clear logs.")
        else:
            print("[*] Clear operation cancelled.")
        sys.exit(0)

    if args.export:
        out_path = Path(args.export)
        try:
            logger.export_csv(out_path)
            print(f"[SUCCESS] CSV log exported successfully to: {out_path}")
        except Exception as e:
            print(f"[ERROR] Failed to export CSV: {e}")
        sys.exit(0)

    # Print summary of inspection records
    stats = logger.get_summary()
    recent = logger.get_recent(args.last)

    print("\n" + "="*50)
    print(" ZERODEFECT INSPECTION LOG SUMMARY")
    print("="*50)
    print(f" Total Inspections : {stats['total']}")
    print(f" OK                : {stats['n_ok']} ({stats['pct_ok']:.1f}%)")
    print(f" DEFECT            : {stats['n_defect']} ({stats['pct_defect']:.1f}%)")
    print("-"*50)
    print(" By Defect Type:")
    if stats["by_defect_type"]:
        for dt, count in stats["by_defect_type"].items():
            print(f"   - {dt:<13} : {count}")
    else:
        print("   (No defects logged yet)")
    print("-"*50)
    print(f" Last Updated      : {stats['last_updated']}")
    print("="*50 + "\n")

    # Table of recent entries
    if recent:
        print(f"Recent Entries (Last {args.last}):")
        print("-"*75)
        print(f" {'ID':<5} | {'TIMESTAMP':<22} | {'DECISION':<8} | {'DEFECT TYPE':<12} | {'CONFIDENCE':<10}")
        print("-"*75)
        for r in recent:
            # Format time slightly shorter
            ts = r['timestamp'].split(".")[0].replace("T", " ")
            def_t = r['defect_type'] if r['defect_type'] else "—"
            conf = f"{r['confidence']*100:.1f}%" if r['confidence'] is not None else "—"
            
            # Print row
            print(f" {r['id']:<5} | {ts:<22} | {r['decision']:<8} | {def_t:<12} | {conf:<10}")
        print("-"*75)
    else:
        print("No recent inspection logs found.")


if __name__ == "__main__":
    main()
