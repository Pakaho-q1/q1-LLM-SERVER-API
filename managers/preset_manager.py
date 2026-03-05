import sqlite3
import json
from typing import Optional, Dict, Any, List


class PresetManager:
    def __init__(self, db_path: str = "data/presets.db"):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS presets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    system_prompt TEXT,
                    parameters TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.commit()

    def create_preset(
        self,
        name: str,
        description: str,
        system_prompt: str,
        parameters: Dict[str, Any],
    ) -> bool:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO presets (name, description, system_prompt, parameters)
                    VALUES (?, ?, ?, ?)
                """,
                    (name, description, system_prompt, json.dumps(parameters)),
                )
                conn.commit()
            return True
        except Exception:
            return False

    def update_preset(
        self,
        name: str,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            with self._connect() as conn:
                preset = self.get_preset(name)
                if not preset:
                    return False

                new_description = (
                    description if description is not None else preset["description"]
                )
                new_prompt = (
                    system_prompt
                    if system_prompt is not None
                    else preset["system_prompt"]
                )
                new_params = (
                    parameters if parameters is not None else preset["parameters"]
                )

                conn.execute(
                    """
                    UPDATE presets
                    SET description = ?, system_prompt = ?, parameters = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE name = ?
                """,
                    (new_description, new_prompt, json.dumps(new_params), name),
                )
                conn.commit()
            return True
        except Exception:
            return False

    def delete_preset(self, name: str) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM presets WHERE name = ?", (name,))
                conn.commit()
            return True
        except Exception:
            return False

    def get_preset(self, name: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT name, description, system_prompt, parameters FROM presets WHERE name = ?",
                (name,),
            )
            row = cur.fetchone()

        if not row:
            return None

        return {
            "name": row[0],
            "description": row[1],
            "system_prompt": row[2],
            "parameters": json.loads(row[3]),
        }

    def list_presets(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.execute("SELECT name, description FROM presets")
            rows = cur.fetchall()

        return [{"name": r[0], "description": r[1]} for r in rows]
