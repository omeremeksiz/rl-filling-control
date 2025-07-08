import psycopg2

class DBHandler:
    def __init__(self, config):
        self.config = config
        self.connection = None

    def connect(self):
        try:
            self.connection = psycopg2.connect(**self.config)
        except Exception as error:
            print(f"Error connecting to database: {error}")
            self.connection = None

    def save_data(self, raw_data, coarse_time, fine_time, total_time, switching_state, overflow_amount, underflow_amount):
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
                self.connection.commit()
                cursor.close()
            except Exception as error:
                print(f"Error saving data to database: {error}")

    def save_statistics(self, valid_fillings, overflow_fillings, underflow_fillings, safe_fillings):
        if self.connection is None:
            self.connect()

        if self.connection:
            try:
                cursor = self.connection.cursor()
                insert_query = """
                INSERT INTO filling_statistics (valid_fillings, overflow_fillings, underflow_fillings, safe_fillings)
                VALUES (%s, %s, %s, %s);
                """
                cursor.execute(insert_query, (valid_fillings, overflow_fillings, underflow_fillings, safe_fillings))
                self.connection.commit()
                cursor.close()
                print("Statistics saved successfully.")
            except Exception as error:
                print(f"Error saving statistics to database: {error}")

    def close_connection(self):
        if self.connection:
            self.connection.close()
            print("Database connection closed.")
