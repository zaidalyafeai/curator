import os
import sqlite3

class MetadataDB():
    def __init__(self, db_path: str):
        self.db_path = db_path

    def store_metadata(self, metadata: dict):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                dataset_hash TEXT,
                user_prompt TEXT,
                system_prompt TEXT,
                model_name TEXT,
                response_format TEXT,
                run_hash TEXT
            )
            ''')
            cursor.execute('''
                INSERT INTO runs (
                    timestamp, dataset_hash, user_prompt, system_prompt,
                    model_name, response_format, run_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata['timestamp'],
                metadata['dataset_hash'],
                metadata['user_prompt'],
                metadata['system_prompt'],
                metadata['model_name'],
                metadata['response_format'],
                metadata['run_hash']
                ))
            conn.commit()