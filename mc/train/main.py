import os
import sys
import yaml
import random
import logging
import datetime
from g_value_handler import GValueHandler
from q_value_handler import QValueHandler
from best_action_handler import BestActionHandler
from file_handler import FileHandler
from plot_handler import PlotHandler
import uuid

def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

def load_config(path="config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)

def main():
    # Load config
    config = load_config()

    # Set seed
    seed = config["training"]["seed"]
    set_seed(seed)

    # Generate a unique job ID and create job folder
    job_id = str(uuid.uuid4())[:8]
    job_output_folder = os.path.join("output", job_id)
    os.makedirs(job_output_folder, exist_ok=True)

    config["output_paths"] = {
        "mc_qvalue_path": os.path.join(job_output_folder, "mc_qvalue_output.txt"),
        "target_state_path": os.path.join(job_output_folder, "target_state.txt"),
        "best_actions_mc_path": os.path.join(job_output_folder, "best_actions.txt"),
        "qvalue_updates_path": os.path.join(job_output_folder, "qvalue_updates.xlsx"),
        "mc_qvalue_html_path": os.path.join(job_output_folder, "mc_qvalue_output.html"),
        "mc_best_actions_html_path": os.path.join(job_output_folder, "best_actions_output.html"),
        "cluster_histogram_path": os.path.join(job_output_folder, "cluster_histogram.png"),
        "qvalue_sate_plot_path": os.path.join(job_output_folder, "qvalue_vs_state.png"),
        "switching_point_trajectory_path": os.path.join(job_output_folder, "switching_point_trajectory.png"),
        "log_file_path": os.path.join(job_output_folder, "training_process.log")
    }

    # Setup logging to the job-specific log file
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(config["output_paths"]["log_file_path"], mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Print job metadata
    logging.info(f"Job ID: {job_id}")
    logging.info(f"Seed: {seed}")
    logging.info(f"All experiment outputs will be saved in: {job_output_folder}\n")

    # Print start time
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"=== Training started at {start_time} ===\n")

    # Print configuration
    logging.info("=== Experiment Configuration ===")
    for section, params in config.items():
        logging.info(f"[{section.upper()}]")
        if isinstance(params, dict):
            for key, value in params.items():
                logging.info(f"{key}: {value}")
        else:
            logging.info(f"{params}")
        logging.info("")

    # Initialize file handler
    file_handler = FileHandler()
    # Initialize plot handler
    plot_handler = PlotHandler(
        output_folder=job_output_folder,
        cluster_histogram_path="cluster_histogram.png",
        qvalue_sate_plot_path="qvalue_vs_state.png",
        switching_point_trajectory_path="switching_point_trajectory.png"
    )

    # Prepare GValueHandler and cluster data by switching points
    g_value_handler = GValueHandler(
        config["training"]["min_weight"], 
        config["training"]["max_weight"],  
        config["training"]["gamma"], 
        config["training"]["overflow_penalty"],
        config["training"]["underflow_penalty"],
        config["data_path"]
    )
    total_training_episodes, max_states_all_episodes = g_value_handler.prepare_clusters() # Prepare clusters and get number of episodes
    
    # Cluster histogram
    plot_handler.plot_cluster_histogram(g_value_handler.clusters)
    logging.info(f"Cluster Histogram plot is saved to: {config['output_paths']['cluster_histogram_path']}\n")

    # Initialize QValueHandler (empty, later updated filling-by-filling)
    q_value_handler = QValueHandler(
        final_arr=None,
        max_states_all_episodes=max_states_all_episodes,
        initial_q_value_value=config["training"]["initial_q_value"],
        target_state_config={
            "weight": config["target_state"]["weight"],
            "action": config["target_state"]["action"]
        },
        learning_rate=config["training"]["learning_rate"],
        max_weight_index=config["training"]["max_weight"]
    )
    
    # Track all episodes info
    all_episodes_info = []

    # Specifying starting switching point
    current_switching_point = 45 # random.choice(list(g_value_handler.clusters.keys()))

    # Start training loop
    for episode_num in range(total_training_episodes):
        logging.info(f"--- Episode {episode_num + 1}/{total_training_episodes} ---")

        # Log the experienced (current) switching point
        experienced_switching_point = current_switching_point

        # Get filling with specific switching point 
        switching_point, selected_filling = g_value_handler.select_filling(current_switching_point)
        if selected_filling is None:
            logging.error(f"No available filling found for current switching point {current_switching_point}.")
            logging.error("No alternative switching point found. Training is terminated.\n")
            break
            
        # Process G-values for selected filling
        termination_type = g_value_handler.process_single_filling(selected_filling)

        # Update Q-values
        q_value_handler.update_q_values(g_value_handler.final_arr)

        # Find best action
        best_action_handler = BestActionHandler(q_value_handler.qValue, q_value_handler.count)
        best_action_handler.find_best_actions()

        # Get best next weight normally (model's selected next switching point)
        next_weight = best_action_handler.get_next_switching_state()
        if next_weight == None:
            next_weight = current_switching_point
        model_selected_next_switching_point = next_weight
        explored_switching_point = None

        if random.random() < config["training"]["exploration_probability"]:
            # Instead of best action, we explore by moving +1, +2, +3, +4 or +5
            offsets = [1, 2, 3, 4, 5]
            weights = [5, 4, 3, 2, 1]
            chosen_offset = random.choices(offsets, weights=weights, k=1)[0]

            next_weight = next_weight + chosen_offset
            explored_switching_point = next_weight

        # Clamp next weight and detect if it was trying to explore out of bounds
        if next_weight > config["training"]["max_weight"]:
            logging.info(f"Warning: Attempted to explore beyond max weight! Clamping from {next_weight} to {config['training']['max_weight']}")
            next_weight = config["training"]["max_weight"]

        # Logging
        logging.info(f"Experienced Switching Point: {experienced_switching_point}")
        logging.info(f"Termination Type: {termination_type}")
        logging.info(f"Model-Selected Next Switching Point: {model_selected_next_switching_point}")
        logging.info(f"Explored Switching Point: {explored_switching_point if explored_switching_point is not None else 'None'}\n")

        # Collect episode info
        all_episodes_info.append({
            "episode_num": episode_num,
            "experienced_switching_point": experienced_switching_point,
            "model_selected_switching_point": model_selected_next_switching_point,
            "explored_switching_point": explored_switching_point,
            "termination_type": termination_type,
            "qValue": q_value_handler.qValue.copy(),
            "count": q_value_handler.count.copy()
        })

        # Update current switching point
        current_switching_point = next_weight

    # After training, save outputs

    # Write final Q-values
    q_value_handler.write_to_text(config["output_paths"]["mc_qvalue_path"])
    logging.info(f"Final Q-values written to: {config['output_paths']['mc_qvalue_path']}")

    # Write Q-values for the target state
    q_value_handler.write_target_state_qvalues_to_text(config["output_paths"]["target_state_path"])
    logging.info(f"Target state Q-values written to: {config['output_paths']['target_state_path']}")

    # Write best actions
    best_action_handler.write_to_text(config["output_paths"]["best_actions_mc_path"])
    logging.info(f"Best actions written to: {config['output_paths']['best_actions_mc_path']}")

    # Write the Excel showing Q-value updates over episodes
    file_handler.write_qvalue_updates_to_excel(
        all_episodes_info,
        config["output_paths"]["qvalue_updates_path"]
    )
    logging.info(f"Q-value updates written to Excel at: {config['output_paths']['qvalue_updates_path']}")    

    # Q-value vs state (best actions)
    plot_handler.plot_qvalue_vs_state(q_value_handler.qValue)
    logging.info(f"Q-Value vs State plot is saved to: {config['output_paths']['qvalue_sate_plot_path']}")
    # Switching point trajectory
    plot_handler.plot_switching_point_trajectory(all_episodes_info)
    logging.info(f"Switching point trajectory_path plot is saved to: {config['output_paths']['switching_point_trajectory_path']}")

    # Format files to HTML
    if config["output_paths"].get("html_control", False):
        file_handler = FileHandler()
        file_handler.format_and_write_table(
            config["output_paths"]["mc_qvalue_path"],
            config["output_paths"]["mc_qvalue_html_path"],
            column_names=['Weight', 'Action', 'Count', 'Q Value']
        )
        logging.info(f"Q-values formatted to HTML at: {config['output_paths']['mc_qvalue_html_path']}")
        file_handler.format_and_write_table(
            config["output_paths"]["best_actions_mc_path"],
            config["output_paths"]["mc_best_actions_html_path"],
            column_names=['Weight', 'Best Action', 'Count', 'Q Value']
        )
        logging.info(f"Best actions formatted to HTML at: {config['output_paths']['mc_best_actions_html_path']}")
    else:
        logging.info(f"HTML feature disabled.")

    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"\n=== Training finished at {end_time} ===")

if __name__ == "__main__":
    main()