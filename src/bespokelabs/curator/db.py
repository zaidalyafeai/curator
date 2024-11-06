"""Database class for storing metadata for Bella runs."""

import os
import sqlite3


class MetadataDB:
    """Database class for storing Bella run metadata."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def store_metadata(self, metadata: dict):
        """Store metadata about a Bella run in the database.

        Args:
            metadata: Dictionary containing run metadata with keys:
                - timestamp: ISO format timestamp
                - dataset_hash: Unique hash of input dataset
                - prompt_func: Source code of prompt function
                - model_name: Name of model used
                - response_format: JSON schema of response format
                - run_hash: Unique hash identifying the run
        """
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    dataset_hash TEXT, 
                    prompt_func TEXT,
                    model_name TEXT,
                    response_format TEXT,
                    run_hash TEXT
                )
                """
            )
            cursor.execute(
                """
                INSERT INTO runs (
                    timestamp, dataset_hash, prompt_func, model_name, response_format, run_hash
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata["timestamp"],
                    metadata["dataset_hash"],
                    metadata["prompt_func"],
                    metadata["model_name"],
                    metadata["response_format"],
                    metadata["run_hash"],
                ),
            )
            conn.commit()
