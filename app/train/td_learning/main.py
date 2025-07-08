import importlib.util
import config

# Dynamically load file_handler module
# file_handler_spec = importlib.util.spec_from_file_location('file_handler', '/Users/omeremeksiz/Desktop/masaustu/RL/application/monte_carlo/RL_imp_v6/file_handler.py')
file_handler_spec = importlib.util.spec_from_file_location('file_handler', '/Users/omeremeksiz/Desktop/masaustu/reinforcement-learning/app/train/monte_carlo/RL_imp_v6/file_handler.py')
file_handler_module = importlib.util.module_from_spec(file_handler_spec)
file_handler_spec.loader.exec_module(file_handler_module)

# Dynamically load state module
# state_spec = importlib.util.spec_from_file_location('state', '/Users/omeremeksiz/Desktop/masaustu/RL/application/monte_carlo/RL_imp_v6/state.py')
state_spec = importlib.util.spec_from_file_location('state', '/Users/omeremeksiz/Desktop/masaustu/reinforcement-learning/app/train/monte_carlo/RL_imp_v6/state.py')
state_module = importlib.util.module_from_spec(state_spec)
state_spec.loader.exec_module(state_module)

from state_initializer import StateInitializer
from q_value_handler import QValueHandler
from best_action_handler import BestActionHandler

def main():
    # Initialize states
    state_initializer = StateInitializer(config.data_path, file_handler_module.FileHandler)
    state_initializer.initialize_states()

    # Write final_arr to text file
    final_arr = state_initializer.get_final_arr()
    file_handler_module.FileHandler().write_final_arr(config.final_arr_path, final_arr)

    # Calculate Q values
    q_value_handler = QValueHandler(final_arr, state_initializer.max_states_per_episode, file_handler_module.FileHandler, state_module.State)

    # Calculate Q values for a single episode
    q_value_handler.calculate_q_values_for_single_episode()
    q_value_handler.write_specific_episode_qvalues_to_text(config.qvalue_update_path)
    q_value_handler.qValue.clear()

    # Calculate Q values for all episodes
    q_value_handler.calculate_q_values_for_all_episodes()
    q_value_handler.write_to_text(config.td_qvalue_path)

    # Write Q values for the target state to a separate file
    q_value_handler.write_target_state_qvalues_to_text(config.qvalue_update_path)

    # Get Q-values and counts
    q_value, count = q_value_handler.get_q_values()

    # Find best actions
    best_action_handler = BestActionHandler(q_value, count, file_handler_module.FileHandler, state_module.State)
    best_action_handler.find_best_actions()

    # Write best actions to text file
    best_action_handler.write_to_text(config.td_best_actions_path)

    # Format the single Q-value file to HTML
    file_handler_module.FileHandler().format_and_write_table(config.td_qvalue_path, config.td_qvalue_html_path, column_names=['Weight', 'Action', 'Count', 'Q Value'])

    # Format the single best action file to HTML
    file_handler_module.FileHandler().format_and_write_table(config.td_best_actions_path, config.td_best_actions_html_path, column_names=['Weight', 'Best Action', 'Count', 'Q Value'])

if __name__ == "__main__":
    main()
