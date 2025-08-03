from data_preprocessing.data_parser import DataParser
from db_handler import DBHandler

class DataPreprocessor:
    def __init__(self, db_handler, quantization_step):
        self.db_handler = db_handler
        self.quantization_step = quantization_step
        self.total_stats = {
            'valid_fillings': 0,
            'overflow_fillings': 0,
            'underflow_fillings': 0,
            'safe_fillings': 0
        }

    def retrieve_and_preprocess_data(self, raw_data_list, tolerance_limits):
        all_parsed_data = []

        for raw_data in raw_data_list:
            parser = DataParser(raw_data, tolerance_limits, quantization_step=self.quantization_step)
            parser.parse_data()

            if parser.is_data_valid():
                parsed_data = parser.get_parsed_data()
                stats = parser.get_stats()

                for key in self.total_stats:
                    self.total_stats[key] += stats[key]

                all_parsed_data.append(parsed_data)

        return all_parsed_data

    def get_total_stats(self):
        return self.total_stats
