"""
logger.py — SQLite-backed thread-safe audit logger with automatic CSV replication.

Provides the InspectionLogger class which saves details, confidence, scores,
and flags thumbnail images for anomalous frames.
"""

import os
import csv
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2

# Default path constants if not imported
ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"
THUMBS_DIR = LOGS_DIR / "thumbnails"


class InspectionLogger:
    """
    Manages inspection audit trails by saving records to SQLite and
    replicating them in a flat CSV file. Saves thumbnail crops of defect parts.
    """
    def __init__(self, log_dir: Path = LOGS_DIR):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.log_dir / "inspections.db"
        self.csv_path = self.log_dir / "inspections.csv"
        self.thumbs_dir = self.log_dir / "thumbnails"
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        
        self.lock = threading.Lock()
        
        # Initialize databases
        self._init_db()
        self._init_csv()

    def _init_db(self):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    image_path TEXT,
                    decision TEXT NOT NULL,
                    defect_type TEXT,
                    confidence REAL,
                    score REAL,
                    source TEXT DEFAULT 'webcam'
                )
            """)
            conn.commit()
            conn.close()

    def _init_csv(self):
        with self.lock:
            if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
                with open(self.csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["id", "timestamp", "image_path", "decision", "defect_type", "confidence", "score", "source"])

    def log(self, decision: str, score: float, confidence: float, defect_type: str = None, 
            image: np.ndarray = None, source: str = "webcam") -> int:
        """
        Write an inspection record to SQLite, append to CSV, and save a thumbnail if an image is provided.
        Returns the inserted row ID.
        """
        timestamp = datetime.now().isoformat()
        row_id = -1
        thumb_filename = ""

        # Thread safe writing
        with self.lock:
            try:
                # 1. Save to SQLite
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO inspections (timestamp, decision, defect_type, confidence, score, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (timestamp, decision, defect_type, confidence, score, source)
                )
                row_id = cursor.lastrowid
                
                # Update image path with index ID if image is provided
                if image is not None:
                    thumb_filename = f"{row_id:04d}_{decision.lower()}"
                    if defect_type:
                        thumb_filename += f"_{defect_type.lower()}"
                    thumb_filename += ".jpg"
                    
                    image_path = f"logs/thumbnails/{thumb_filename}"
                    cursor.execute(
                        "UPDATE inspections SET image_path = ? WHERE id = ?",
                        (image_path, row_id)
                    )
                else:
                    image_path = ""
                
                conn.commit()
                conn.close()

                # 2. Save thumbnail image to disk
                if image is not None and thumb_filename:
                    # Save a 224x224 thumbnail representation
                    thumb = cv2.resize(image, (224, 224))
                    cv2.imwrite(str(self.thumbs_dir / thumb_filename), thumb)

                # 3. Append to CSV
                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([row_id, timestamp, image_path, decision, defect_type or "", confidence, score, source])
                    
            except Exception as e:
                print(f"[WARN] Logging failed: {e}")
                
        return row_id

    def get_summary(self) -> dict:
        """Query SQLite database for aggregate counts and stats."""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Total count
                cursor.execute("SELECT COUNT(*) FROM inspections")
                total = cursor.fetchone()[0]
                
                # OK counts
                cursor.execute("SELECT COUNT(*) FROM inspections WHERE decision = 'OK'")
                n_ok = cursor.fetchone()[0]
                
                # Defect counts
                cursor.execute("SELECT COUNT(*) FROM inspections WHERE decision = 'DEFECT'")
                n_defect = cursor.fetchone()[0]
                
                # Breakdown
                cursor.execute("""
                    SELECT defect_type, COUNT(*) 
                    FROM inspections 
                    WHERE decision = 'DEFECT' AND defect_type IS NOT NULL 
                    GROUP BY defect_type
                """)
                by_defect_type = {}
                for row in cursor.fetchall():
                    by_defect_type[row[0]] = row[1]
                
                cursor.execute("SELECT MAX(timestamp) FROM inspections")
                last_updated = cursor.fetchone()[0] or "Never"
                
                conn.close()
                
                pct_ok = (n_ok / total * 100) if total > 0 else 0.0
                pct_defect = (n_defect / total * 100) if total > 0 else 0.0
                
                return {
                    "total": total,
                    "n_ok": n_ok,
                    "n_defect": n_defect,
                    "pct_ok": pct_ok,
                    "pct_defect": pct_defect,
                    "by_defect_type": by_defect_type,
                    "last_updated": last_updated
                }
            except Exception as e:
                print(f"[ERROR] Failed to fetch summary: {e}")
                return {
                    "total": 0, "n_ok": 0, "n_defect": 0,
                    "pct_ok": 0, "pct_defect": 0, "by_defect_type": {},
                    "last_updated": "Error"
                }

    def get_recent(self, n: int = 10) -> list:
        """Return the last N records."""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM inspections ORDER BY id DESC LIMIT ?", (n,)
                )
                rows = [dict(row) for row in cursor.fetchall()]
                conn.close()
                return rows
            except Exception as e:
                print(f"[ERROR] Failed to fetch recent logs: {e}")
                return []

    def clear(self) -> bool:
        """Truncate all records in the SQLite database and CSV file."""
        with self.lock:
            try:
                # Clear SQLite
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM inspections")
                cursor.execute("DELETE FROM sqlite_sequence WHERE name='inspections'")
                conn.commit()
                conn.close()
                
                # Truncate CSV
                with open(self.csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["id", "timestamp", "image_path", "decision", "defect_type", "confidence", "score", "source"])
                
                # Delete thumbnails
                for thumb in self.thumbs_dir.glob("*.jpg"):
                    try:
                        thumb.unlink()
                    except OSError:
                        pass
                return True
            except Exception as e:
                print(f"[ERROR] Failed to clear logs: {e}")
                return False

    def export_csv(self, output_path: str | Path) -> None:
        """Copy the csv log to an output path."""
        with self.lock:
            import shutil
            shutil.copy(self.csv_path, output_path)
