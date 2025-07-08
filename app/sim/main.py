import yaml
import os
import logging
import sys
import datetime
import pprint

from com.tcp_com import TCPClient
from com.modbus_com import ModbusClient
from data_preprocessing.data_preprocessor import DataPreprocessor
from db_handler import DBHandler
from data_preprocessing.data_parser import DataParser
from model.g_value_handler import GValueHandler
from model.q_value_handler import QValueHandler
from model.best_action_handler import BestActionHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  
    handlers=[
        logging.FileHandler("filling_process.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

def load_config(path="config.yaml"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, path)
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

class FillingProcess:
    def __init__(self, config):
        self.config = config
        self.rl_config = config["rl_config"]
        self.tcp_client = TCPClient(config["network"]["tcp_ip"], config["network"]["tcp_port"])
        self.modbus_client = ModbusClient(config["network"]["modbus_ip"], config["network"]["modbus_port"])
        self.db_handler = DBHandler({
            "database": config["database"]["name"],
            "user": config["database"]["user"],
            "password": config["database"]["password"],
            "host": config["database"]["host"],
            "port": str(config["database"]["port"])
        })
        self.preprocessor = DataPreprocessor(db_handler=self.db_handler, quantization_step=self.rl_config["weight_quantization_step"])
        self.tolerance_limits = tuple(config["filling"]["tolerance_limits"])
        self.modbus_register = config["filling"]["modbus_register"]
        self.q_output_paths = config["output_paths"]
    
    def run(self):
        self.tcp_client.connect()
        self.modbus_client.connect()

        try:
            total_stats = {'valid_fillings': 0, 'overflow_fillings': 0, 'underflow_fillings': 0, 'safe_fillings': 0}
            while True:
                raw_data = self.tcp_client.receive_data()
                if raw_data is None:
                    logging.info("No data received. Waiting for next filling episode...")
                    continue

                logging.info(f"Received raw data: {raw_data}")

                parser = DataParser(raw_data, self.tolerance_limits, quantization_step=self.rl_config["weight_quantization_step"])
                parser.parse_data(is_original=True)
                original_id = self.db_handler.save_original_data(
                    raw_data,
                    parser.coarse_time,
                    parser.fine_time,
                    parser.total_time,
                    parser.switching_state,
                    parser.overflow_amount,
                    parser.underflow_amount
                )

                parser.parse_data(is_original=False)
                parsed_id = self.db_handler.save_parsed_data(
                    original_id,
                    parser.parsed_data,
                    parser.coarse_time,
                    parser.fine_time,
                    parser.total_time,
                    parser.switching_state,
                    parser.overflow_amount,
                    parser.underflow_amount
                )

                query = f"SELECT raw_data FROM parsed_filling_data_table WHERE id = {parsed_id}"
                g_value_handler = GValueHandler(
                    self.db_handler, 
                    query, 
                    self.tolerance_limits,
                    overflow_scale=self.rl_config["penalties"]["overflow_scale"],
                    underflow_scale=self.rl_config["penalties"]["underflow_scale"],
                    gama=self.rl_config["gamma"]
                )
                g_value_handler.process_g_values()

                last_q_value = self.db_handler.get_last_q_value(current_row_id=parsed_id)

                q_value_handler = QValueHandler(
                    final_arr=g_value_handler.final_arr,
                    max_states_per_episode=g_value_handler.max_states_per_episode,
                    upper_tolerance_weight=self.tolerance_limits[1],
                    learning_rate=self.rl_config["learning_rate"],
                    initial_q_value_value=self.rl_config["initial_q_value"],
                    target_state_config=self.rl_config["target_state"],
                    initial_q_value=last_q_value
                )
                q_value_handler.calculate_q_values()

                # Use State objects for q_value
                self.db_handler.update_q_value(
                    row_id=parsed_id,
                    q_value={state: (q_value_handler.count[state], q_val)
                            for state, q_val in q_value_handler.qValue.items()}
                )

                # Output paths for q_value_handler results
                q_value_handler.write_to_text(self.q_output_paths["q_value_output"])
                q_value_handler.write_target_state_qvalues_to_text(self.q_output_paths["q_target_output"])
                logging.info(f"Q values written to: {self.q_output_paths['q_value_output']}")
                logging.info(f"Target state Q values written to: {self.q_output_paths['q_target_output']}")

                best_action_handler = BestActionHandler(q_value_handler.qValue, q_value_handler.count)
                best_action_handler.find_best_actions()
                flip_weight = best_action_handler.find_action_flip()
                chosen_weight = best_action_handler.exploration_or_explotation(
                    flip_weight=flip_weight, prob=self.rl_config["exploration_probability"]
                )
                if flip_weight != chosen_weight: logging.info(f"Exploration: Previous weight: {flip_weight}, Chosen weight: {chosen_weight}")
                else: logging.info(f"Explotation: Previous weight: {flip_weight}, Chosen weight: {chosen_weight}")

                self.modbus_client.send_weight(register=self.modbus_register, weight=chosen_weight)  # Register 40001
                
                # Output path for best_action_handler results
                best_action_handler.write_to_text(self.q_output_paths["best_action_output"])
                logging.info(f"Best actions written to: {self.q_output_paths['best_action_output']}")

                filling_stats = parser.get_stats()
                for key in total_stats:
                    total_stats[key] += filling_stats[key]

        finally:
            self.db_handler.save_statistics(**total_stats)
            self.tcp_client.close_connection()
            self.modbus_client.close()
            self.db_handler.close_connection()

if __name__ == "__main__":
    config = load_config()

    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"=== Experiment started at {start_time} ===\n")
    logging.info("=== Experiment Configuration ===")
    for section, params in config.items():
        logging.info(f"[{section.upper()}]")
        for key, value in params.items():
            logging.info(f"{key}: {value}")
        logging.info("")

    filling_process = FillingProcess(config)
    filling_process.run()
