"""Database utilities for saving real-world filling episodes."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import mysql.connector
from mysql.connector import Error

DEFAULT_DB_CONFIG: Dict[str, Any] = {
    "name": "filling_data",
    "user": "root",
    "password": "6637",
    "host": "127.0.0.1",
    "port": 3306,
}

class DatabaseHandler:
    """Handles database operations for real-world filling episodes."""

    def __init__(self, db_config: Optional[Dict[str, Any]] = None) -> None:
        merged_config = {**DEFAULT_DB_CONFIG, **(db_config or {})}
        if isinstance(merged_config.get("port"), str):
            try:
                merged_config["port"] = int(merged_config["port"])
            except ValueError:
                merged_config["port"] = DEFAULT_DB_CONFIG["port"]
        self.config: Dict[str, Any] = merged_config
        self.connection = None

    def connect(self) -> bool:

        try:
            self.connection = mysql.connector.connect(
                database=self.config["name"],
                user=self.config["user"],
                password=self.config["password"],
                host=self.config["host"],
                port=self.config["port"],
            )
            logging.info("Database connection successful")
            self._create_tables_if_not_exist()
            return True
        except Error as exc:
            logging.error(f"Error connecting to database: {exc}")
            self.connection = None
            return False

    def _create_tables_if_not_exist(self) -> None:
        if not self.connection:
            return

        try:
            cursor = self.connection.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS filling_data_table (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    raw_data TEXT,
                    coarse_time INT,
                    fine_time INT,
                    total_time INT,
                    switching_state INT,
                    overflow_amount INT DEFAULT 0,
                    underflow_amount INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS parsed_filling_data_table (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    original_id INT,
                    raw_data TEXT,
                    coarse_time INT,
                    fine_time INT,
                    total_time INT,
                    switching_state INT,
                    overflow_amount INT DEFAULT 0,
                    underflow_amount INT DEFAULT 0,
                    q_value JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (original_id) REFERENCES filling_data_table(id)
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS filling_statistics_table (
                    id INT PRIMARY KEY,
                    valid_fillings INT DEFAULT 0,
                    overflow_fillings INT DEFAULT 0,
                    underflow_fillings INT DEFAULT 0,
                    safe_fillings INT DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                )
                """
            )

            self.connection.commit()
            cursor.close()
            logging.info("Database tables verified/created")
        except Error as exc:
            logging.error(f"Error creating tables: {exc}")

    def save_original_episode(
        self,
        raw_data: str,
        coarse_time: int,
        fine_time: int,
        total_time: int,
        switching_state: int,
        overflow_amount: int = 0,
        underflow_amount: int = 0,
    ) -> Optional[int]:
        if not self.connection and not self.connect():
            return None

        try:
            cursor = self.connection.cursor()
            query = (
                "INSERT INTO filling_data_table "
                "(raw_data, coarse_time, fine_time, total_time, switching_state, overflow_amount, underflow_amount)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)"
            )
            cursor.execute(
                query,
                (
                    raw_data,
                    coarse_time,
                    fine_time,
                    total_time,
                    switching_state,
                    overflow_amount,
                    underflow_amount,
                ),
            )
            row_id = cursor.lastrowid
            self.connection.commit()
            cursor.close()
            logging.info(f"Original episode data saved with ID: {row_id}")
            return row_id
        except Error as exc:
            logging.error(f"Error saving original episode: {exc}")
            return None

    def save_parsed_episode(
        self,
        original_id: int,
        weight_sequence: List[int],
        coarse_time: int,
        fine_time: int,
        total_time: int,
        switching_state: int,
        overflow_amount: int = 0,
        underflow_amount: int = 0,
    ) -> Optional[int]:
        if not self.connection and not self.connect():
            return None

        try:
            raw_data = ",".join(map(str, weight_sequence)) if isinstance(weight_sequence, list) else str(weight_sequence)
            cursor = self.connection.cursor()
            query = (
                "INSERT INTO parsed_filling_data_table "
                "(original_id, raw_data, coarse_time, fine_time, total_time, switching_state, overflow_amount, underflow_amount)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            )
            cursor.execute(
                query,
                (
                    original_id,
                    raw_data,
                    coarse_time,
                    fine_time,
                    total_time,
                    switching_state,
                    overflow_amount,
                    underflow_amount,
                ),
            )
            row_id = cursor.lastrowid
            self.connection.commit()
            cursor.close()
            logging.info(f"Parsed episode data saved with ID: {row_id}")
            return row_id
        except Error as exc:
            logging.error(f"Error saving parsed episode: {exc}")
            return None

    def update_q_values(self, episode_id: int, q_values: Dict[Any, Any]) -> bool:
        if not self.connection and not self.connect():
            return False

        try:
            serialized_q_values = json.dumps({str(key): value for key, value in q_values.items()})
            cursor = self.connection.cursor()
            query = "UPDATE parsed_filling_data_table SET q_value = %s WHERE id = %s"
            cursor.execute(query, (serialized_q_values, episode_id))
            self.connection.commit()
            cursor.close()
            logging.info(f"Q-values updated for episode {episode_id}")
            return True
        except Error as exc:
            logging.error(f"Error updating Q-values: {exc}")
            return False

    def save_statistics(self, **stats: int) -> bool:
        if not self.connection and not self.connect():
            return False

        stats_defaults = {
            "valid_fillings": 0,
            "overflow_fillings": 0,
            "underflow_fillings": 0,
            "safe_fillings": 0,
        }
        stats_payload = {**stats_defaults, **stats}

        try:
            cursor = self.connection.cursor()
            query = (
                "INSERT INTO filling_statistics_table "
                "(id, valid_fillings, overflow_fillings, underflow_fillings, safe_fillings) "
                "VALUES (1, %(valid_fillings)s, %(overflow_fillings)s, %(underflow_fillings)s, %(safe_fillings)s) "
                "ON DUPLICATE KEY UPDATE "
                "valid_fillings = valid_fillings + VALUES(valid_fillings), "
                "overflow_fillings = overflow_fillings + VALUES(overflow_fillings), "
                "underflow_fillings = underflow_fillings + VALUES(underflow_fillings), "
                "safe_fillings = safe_fillings + VALUES(safe_fillings)"
            )
            cursor.execute(query, stats_payload)
            self.connection.commit()
            cursor.close()
            logging.info("Statistics updated successfully")
            return True
        except Error as exc:
            logging.error(f"Error saving statistics: {exc}")
            return False

    def get_episode_data(self, episode_id: int) -> Optional[Dict[str, Any]]:
        if not self.connection and not self.connect():
            return None

        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM parsed_filling_data_table WHERE id = %s", (episode_id,))
            result = cursor.fetchone()
            cursor.close()
            return result
        except Error as exc:
            logging.error(f"Error retrieving episode data: {exc}")
            return None

    def close(self) -> None:
        if self.connection:
            self.connection.close()
            self.connection = None
            logging.info("Database connection closed")
