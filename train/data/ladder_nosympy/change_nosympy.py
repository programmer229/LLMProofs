import json
import pandas as pd
import os

def convert_json_to_parquet(input_json_path, output_parquet_path, output_json_path=None, change_data_source=False):
    """
    Convert JSON file to Parquet format, optionally changing the data_source field.
    
    Args:
        input_json_path: Path to the input JSON file
        output_parquet_path: Path to save the output Parquet file
        output_json_path: Path to save the modified JSON file (optional)
        change_data_source: Whether to change the data_source field to "llm_judge_integration"
    """
    # Read the JSON file
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Change data_source if needed
    if change_data_source:
        for item in data:
            item['data_source'] = "llm_judge_integration"
    
    # Save modified JSON if path is provided
    if output_json_path:
        with open(output_json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved modified JSON to {output_json_path}")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save as Parquet
    df.to_parquet(output_parquet_path, index=False)
    print(f"Converted {input_json_path} to {output_parquet_path}")

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
old_dir = "/home/ubuntu/o1-replication-sydney/CustomTinyZero/data/ladder_sympy"

train_json_path = os.path.join(old_dir, 'train.json')
test_json_path = os.path.join(old_dir, 'test.json')
train_parquet_path = os.path.join(current_dir, 'train.parquet')
test_parquet_path = os.path.join(current_dir, 'test.parquet')
train_json_output_path = os.path.join(current_dir, 'train.json')
test_json_output_path = os.path.join(current_dir, 'test.json')

# Convert train.json with data_source change
convert_json_to_parquet(train_json_path, train_parquet_path, train_json_output_path, change_data_source=True)

# Convert test.json without data_source change
convert_json_to_parquet(test_json_path, test_parquet_path, test_json_output_path, change_data_source=False)

print("Conversion complete!")
