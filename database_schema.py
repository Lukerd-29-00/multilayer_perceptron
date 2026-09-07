import sqlite3
import os


database_file = os.environ.get("TRAINING_DATA_FILE")

assert database_file is not None

with sqlite3.connect(database_file) as conn:

    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS TRAINING_RUN (
        id INTEGER PRIMARY KEY,
        learnset TEXT NOT NULL
    )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_training_run_learnset ON TRAINING_RUN(learnset)
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS TRAINING_RUN_DATA (
            input_or_output TEXT NOT NULL,
            training_run INTEGER NOT NULL,
            idx INTEGER NOT NULL,
            value REAL NOT NULL,
            PRIMARY KEY(input_or_output, training_run, idx)
        )""")