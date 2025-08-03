import mysql.connector
import pandas as pd
import json
import logging
from model.state import State
from mysql.connector import Error

class DBHandler:
    def __init__(self, config):
        self.config = config 
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
                host=self.config['host'],
                port=self.config['port']
            )
            logging.info("Database connection successful!")
        except Error as error:
            logging.error(f"Error connecting to database: {error}")
            self.connection = None

    def save_original_data(self, raw_data, coarse_time, fine_time, total_time, switching_state, overflow_amount, underflow_amount):
        if self.connection is None:
            self.connect()

        if self.connection:
            try:
                cursor = self.connection.cursor()
                insert_query = """
                INSERT INTO filling_data_table 
                (raw_data, coarse_time, fine_time, total_time, switching_state, overflow_amount, underflow_amount)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
                """
                cursor.execute(insert_query, (raw_data, coarse_time, fine_time, total_time, switching_state, overflow_amount, underflow_amount))
                row_id = cursor.lastrowid
                self.connection.commit()
                cursor.close()
                logging.info("Original data saved successfully.")
                return row_id
            except Exception as error:
                logging.error(f"Error saving original data to database: {error}")

    def save_parsed_data(self, id, raw_data, coarse_time, fine_time, total_time, switching_state, overflow_amount, underflow_amount):
        if self.connection is None:
            self.connect()

        if isinstance(raw_data, list):
            raw_data = ','.join(map(str, raw_data))

        if self.connection:
            try:
                cursor = self.connection.cursor()
                insert_query = """
                INSERT INTO parsed_filling_data_table 
                (id, raw_data, coarse_time, fine_time, total_time, switching_state, overflow_amount, underflow_amount)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                """
                cursor.execute(insert_query, (id, raw_data, coarse_time, fine_time, total_time, switching_state, overflow_amount, underflow_amount))
                row_id = cursor.lastrowid
                self.connection.commit()
                cursor.close()
                logging.info("Parsed data saved successfully.")
                return row_id
            except Exception as error:
                logging.error(f"Error saving parsed data to database: {error}") 
                return None

    def update_q_value(self, row_id, q_value):
        if self.connection is None:
            self.connect()

        if self.connection:
            try:
                cursor = self.connection.cursor()
                serialized_q_value = json.dumps({
                    f"{state.weight},{state.action}": [count, q_val]
                    for state, (count, q_val) in q_value.items()
                })
                update_query = """
                UPDATE parsed_filling_data_table
                SET q_value = %s
                WHERE id = %s;
                """
                cursor.execute(update_query, (serialized_q_value, row_id))
                self.connection.commit()
                cursor.close()
                logging.info(f"Q-Value updated for row {row_id}.")
            except Exception as error:
                logging.error(f"Error updating Q-Value in database: {error}")

    def get_last_q_value(self, current_row_id):
        if self.connection is None:
            self.connect()

        if self.connection:
            try:
                cursor = self.connection.cursor()
                query = """
                SELECT q_value
                FROM parsed_filling_data_table
                WHERE id < %s
                ORDER BY id DESC
                LIMIT 1;
                """
                cursor.execute(query, (current_row_id,))
                result = cursor.fetchone()
                cursor.close()

                if result and result[0]:
                    q_value_dict = json.loads(result[0])
                    return {
                        State(*map(float, key.split(","))): (count, q_val)
                        for key, (count, q_val) in q_value_dict.items()
                    }
                else:
                    return {}
            except Exception as error:
                logging.error(f"Error retrieving Q-Value from database: {error}")
                return {}

    def save_statistics(self, **stats):
        if self.connection is None:
            self.connect()

        if self.connection:
            try:
                cursor = self.connection.cursor()

                insert_update_query = """
                INSERT INTO filling_statistics_table 
                (id, valid_fillings, overflow_fillings, underflow_fillings, safe_fillings)
                VALUES (1, %(valid_fillings)s, %(overflow_fillings)s, %(underflow_fillings)s, %(safe_fillings)s)
                ON DUPLICATE KEY UPDATE 
                valid_fillings = valid_fillings + VALUES(valid_fillings),
                overflow_fillings = overflow_fillings + VALUES(overflow_fillings),
                underflow_fillings = underflow_fillings + VALUES(underflow_fillings),
                safe_fillings = safe_fillings + VALUES(safe_fillings);
                """
                cursor.execute(insert_update_query, stats)
                self.connection.commit()
                cursor.close()
                logging.info("Statistics updated successfully.")
            except Exception as error:
                logging.error(f"Error updating statistics in database: {error}")

    def fetch_data(self, query):
        if self.connection is None:
            self.connect()

        if self.connection:
            try:
                cursor = self.connection.cursor(dictionary=True)
                cursor.execute(query)
                result = cursor.fetchall()
                cursor.close()
                
                df = pd.DataFrame(result)
                return df
            except Exception as error:
                logging.error(f"Error fetching data from database: {error}")
                return None

    def close_connection(self):
        if self.connection:
            self.connection.close()
            logging.info("Database connection closed.")
