import re
from os.path import dirname, join, abspath
import pandas as pd

CSV_FILE = 'df_train_tx.csv'
OUT_FILES = ['nohup_1.out', 'nohup_2.out']


#################### Get the current directory
curr_dir = dirname(abspath(__file__))

# Load the CSV file
csv_file = join(curr_dir, CSV_FILE)   # Replace with your CSV file path
df = pd.read_csv(csv_file)

# Regular expression pattern to extract 'number' and 'elapsed' values
pattern = r"number=(\d+) .* elapsed=([\d\.]+)"
elapsed_times = {}

for out_file in OUT_FILES:
    file_path = join(curr_dir, join(curr_dir, out_file)) 
    with open(file_path, 'r') as file:  # Open the file and process each line
        for line in file:
            if 'Successfully sealed new block' in line:
                match = re.search(pattern, line)

                if match:
                    number = int(match.group(1))
                    elapsed = float(match.group(2))
                    if elapsed > 100:
                        elapsed = round(elapsed/100, 4)
                    print(f"Number: {number}, Elapsed: {elapsed}")
                    elapsed_times[number] = elapsed
                else:
                    print("Pattern not found in the string.")
                    print(line)
                
print(elapsed_times)



# Map the 'block_num' column to the 'elapsed' column using the dictionary
df['elapsed'] = df['block_num'].map(elapsed_times)

# Save the updated DataFrame back to a CSV file
output_file = f"{CSV_FILE.split('.')[0]}_plus.csv"  # Replace with your desired output file path
df.to_csv(output_file, index=False)