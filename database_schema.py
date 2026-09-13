import sqlite3
import argparse

parser = argparse.ArgumentParser()

file = parser.add_argument("db_file")

args = parser.parse_args()

database_file = args.db_file

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