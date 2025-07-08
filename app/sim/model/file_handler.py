import pandas as pd
import webbrowser
import os

class FileHandler:
    def write_to_text(self, file_path, header, data):
        try:
            with open(file_path, "w") as f:
                print(header, file=f)
                for item in data:
                    print(item, file=f)
        except Exception as e:
            print(f"Error writing to text file: {e}")

    def append_to_text(self, file_path, data):
        try:
            with open(file_path, "a") as f:
                for item in data:
                    print(item, file=f)
        except Exception as e:
            print(f"Error appending to text file: {e}")

    def read_txt(self, file_path, column_names=None):
        try:
            data = pd.read_csv(file_path, delimiter='\t', skiprows=1, header=None)
            if column_names:
                data.columns = column_names
            else:
                data.columns = ['Weight', 'Action', 'Count', 'Q Value']
            return data
        except Exception as e:
            print(f"Error reading txt file: {e}")
            return None

    def write_tolerance_pairs(self, file_path, tolerance_pairs):
        sorted_pairs = sorted(tolerance_pairs.items(), key=lambda x: x[0].cutoff_weight)
        
        tolerance_output = []
        for pair, weight_margin in sorted_pairs:
            tolerance_output.append(f"{pair.cutoff_weight}\t{pair.error_type}\t{weight_margin}")

        header = "Cutoff Weight\tError Type\tWeight Margin"
        self.write_to_text(file_path, header, tolerance_output)

    def write_final_arr(self, file_path, final_arr):
        final_output = []

        for state in final_arr[5, :]:
            weight, action, reward = state
            final_output.append(f"{weight}\t{int(action)}\t{reward}")
        
        header = "Weight\tAction\tG Value"
        self.write_to_text(file_path, header, final_output)

    def write_q_values(self, file_path, q_values, count):
        q_value_output = []
        sorted_items = sorted(q_values.items(), key=lambda x: (x[0].weight, x[0].action))
        for state, reward in sorted_items:
            count_value = count[state]
            q_value_output.append(f"{state.weight}\t{int(state.action)}\t{count_value}\t{reward}")
        
        header = "Weight\tAction\tCount\tQ Value"
        self.write_to_text(file_path, header, q_value_output)

    def write_target_state_qvalues(self, file_path, q_values_for_target_state):
        q_value_output = []
        for i, (q_value, difference, reward) in enumerate(q_values_for_target_state, start=1):
            q_value_output.append(f"{i}\t{q_value}\t{difference}\t{reward}")

        header = "Iteration\tQ Value\tDifference\tG Value"
        self.write_to_text(file_path, header, q_value_output)

    def write_best_actions(self, file_path, best_action):
        best_actions_output = []
        for weight, (action, q_value, count_value) in best_action.items():
            best_actions_output.append(f"{weight}\t{action}\t{count_value}\t{q_value}")
        
        header = "Weight\tBest Action\tCount\tQ Value"
        self.write_to_text(file_path, header, best_actions_output)

    def format_and_write_table(self, input_file_path, output_html_path, column_names=None):
        df = self.read_txt(input_file_path, column_names)
        if df is not None:
            try:
                df.to_html(output_html_path, index=False, border=1, justify='center')
                print(f"Formatted table written to {output_html_path}")
                self.open_in_browser(output_html_path)
            except Exception as e:
                print(f"Error writing HTML file: {e}")
        else:
            print("Failed to read and format the table. Please check the input file.")


    def merge_qvalue_files(self, mc_file_path, td_file_path, ql_file_path, output_html_path):
        try:
            mc_df = pd.read_csv(mc_file_path, delimiter='\t', skiprows=1, header=None, names=['Weight', 'Action', 'MC Count', 'MC Q Value'])
            td_df = pd.read_csv(td_file_path, delimiter='\t', skiprows=1, header=None, names=['Weight', 'Action', 'TD Count', 'TD Q Value'])
            ql_df = pd.read_csv(ql_file_path, delimiter='\t', skiprows=1, header=None, names=['Weight', 'Action', 'QL Count', 'QL Q Value'])

            merged_df = pd.merge(mc_df, td_df, on=['Weight', 'Action'])
            merged_df = pd.merge(merged_df, ql_df, on=['Weight', 'Action'])
            merged_df = merged_df[['Weight', 'Action', 'MC Count', 'MC Q Value', 'TD Count', 'TD Q Value', 'QL Count', 'QL Q Value']]
            
            merged_df.to_html(output_html_path, index=False, border=1, justify='center')
            print(f"Merged table written to {output_html_path}")
            self.open_in_browser(output_html_path)
        except Exception as e:
            print(f"Error merging Q-value files: {e}")

    def merged_best_actions(self, mc_file_path, td_file_path, ql_file_path, output_html_path):
        try:
            mc_df = self.read_txt(mc_file_path, column_names=['Weight', 'MC Best Action', 'MC Count', 'MC Q Value'])
            td_df = self.read_txt(td_file_path, column_names=['Weight', 'TD Best Action', 'TD Count', 'TD Q Value'])
            ql_df = self.read_txt(ql_file_path, column_names=['Weight', 'QL Best Action', 'QL Count', 'QL Q Value'])

            if mc_df is None or td_df is None or ql_df is None:
                print("Failed to read input files. Please check the file paths.")
                return

            merged_df = pd.merge(mc_df, td_df, on='Weight')
            merged_df = pd.merge(merged_df, ql_df, on='Weight')
            merged_df = merged_df[['Weight', 'MC Best Action', 'MC Count', 'MC Q Value', 'TD Best Action', 'TD Count', 'TD Q Value', 'QL Best Action', 'QL Count', 'QL Q Value']]
            
            merged_df.to_html(output_html_path, index=False, border=1, justify='center')
            print(f"Merged best actions table written to {output_html_path}")
            self.open_in_browser(output_html_path)
        except Exception as e:
            print(f"Error merging best actions files: {e}")


    def open_in_browser(self, file_path):
        try:
            if os.path.exists(file_path):
                webbrowser.open(f'file://{os.path.realpath(file_path)}', new=2)
            else:
                print("HTML file not found.")
        except Exception as e:
            print(f"Error opening HTML file in browser: {e}")
