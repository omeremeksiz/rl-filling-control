from db_handler import DBHandler
from excel_precessor import ExcelProcessor

def main():
    db_config = {
        'database': 'filling_data',
        'user': 'postgres',
        'password': '6637',
        'host': '127.0.0.1',
        'port': '5432'
    }

    excel_file_path = "C:/Users/omer.emeksiz/Desktop/masaustu/reinforcement-learning/data/data_v3/filling_data/dolum_verileri_yeni.xlsx"
    tolerance_limits = (740000, 760000)  # Example tolerance limits, adjust as needed

    db_handler = DBHandler(db_config)
    db_handler.connect()

    excel_processor = ExcelProcessor(excel_file_path, tolerance_limits)
    processed_data, valid_fillings, overflow_fillings, underflow_fillings, safe_fillings = excel_processor.process_data()

    try:
        for data in processed_data:
            db_handler.save_data(*data)
        
        # Save statistics to the database
        db_handler.save_statistics(valid_fillings, overflow_fillings, underflow_fillings, safe_fillings)

        print("All data from the Excel file and statistics saved.")
    finally:
        db_handler.close_connection()

if __name__ == "__main__":
    main()
