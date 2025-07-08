import pandas as pd

class ExcelProcessor:
    def __init__(self, excel_file_path, tolerance_limits):
        self.excel_file_path = excel_file_path
        self.tolerance_limits = tolerance_limits

    def process_data(self):
        df = pd.read_excel(self.excel_file_path, header=None)
        processed_data = []
        valid_fillings = 0
        overflow_fillings = 0
        underflow_fillings = 0
        safe_fillings = 0

        for index, row in df.iterrows():
            row_data = str(row.iloc[0])[29:]
            row_data = row_data.replace(" ", "")
            pairs = row_data.split(';')
            pairs = [pair for pair in pairs if pair]

            if not pairs:
                continue

            if "300,300" not in pairs:
                print(f"Skipping row {index + 1}: No '300,300' pair found.")
                continue

            if "30,30" not in pairs:
                print(f"Skipping row {index + 1}: No '30,30' pair found.")
                continue

            first_pair_first_element = int(pairs[0].split(',')[0])
            if first_pair_first_element >= 5000:
                print(f"Skipping row {index + 1}: First element of first pair is greater than 5000.")
                continue

            last_pair = pairs[-1]
            try:
                fine_time, total_time = map(int, last_pair.split(','))
            except ValueError:
                continue

            fine_time = fine_time * 100
            total_time = total_time * 10
            coarse_time = total_time - fine_time

            switching_state = None
            for i in range(len(pairs) - 1, -1, -1):
                if pairs[i] == "30,30":
                    if i > 0:
                        switching_state = int(pairs[i - 1].split(',')[0])
                    break

            overflow_amount, underflow_amount = 0, 0
            for i in range(len(pairs)):
                if pairs[i] == "300,300":
                    if i + 1 < len(pairs):
                        total_weight = int(pairs[i + 1].split(',')[0])
                        if total_weight > self.tolerance_limits[1]:
                            overflow_fillings += 1
                            overflow_amount = total_weight - self.tolerance_limits[1]
                        elif total_weight < self.tolerance_limits[0]:
                            underflow_fillings += 1
                            underflow_amount = self.tolerance_limits[0] - total_weight
                        else:
                            safe_fillings += 1
                    break

            processed_data.append((row_data, coarse_time, fine_time, total_time, switching_state, overflow_amount, underflow_amount))
            valid_fillings += 1

        print(f"Number of valid fillings: {valid_fillings}")
        print(f"Number of overflow fillings: {overflow_fillings}")
        print(f"Number of underflow fillings: {underflow_fillings}")
        print(f"Number of safe fillings: {safe_fillings}")

        return processed_data, valid_fillings, overflow_fillings, underflow_fillings, safe_fillings
