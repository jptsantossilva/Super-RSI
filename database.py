import sqlite3
import os
import pandas as pd

def connect(path=""):
    file_path = os.path.join(path, "super-rsi.db")
    return sqlite3.connect(file_path)

# connection = connect()

sql_get_positions_by_bot_position = """
    SELECT *
    FROM Positions 
    WHERE 
        Position = ?
"""
def get_positions_by_bot_position(connection, position):
    return pd.read_sql(sql_get_positions_by_bot_position, connection, params=(position,))