# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the Lichess puzzle dataset to parquet format
"""

import os
import datasets
import argparse

def create_puzzle_prompt(board_repr, puzzle_data):
    """Create a formatted prompt for a chess puzzle.
    
    Args:
        board_repr (str): String representation of the chess board
        puzzle_data (dict): Dictionary containing puzzle metadata
        
    Returns:
        str: Formatted prompt string
    """
    return f"""Here's a chess puzzle. Find the next move to solve the puzzle.\n\nPosition:\n{board_repr}\n\nFEN: {puzzle_data['FEN']}\nRating: {puzzle_data['Rating']}\nThemes: {puzzle_data['Themes']}
            
        Provide ONLY the next move in algebraic notation within answer tags, like:
        <answer>e4e5</answer>

    """

def create_board_representation(fen):
    """Create a string representation of the chess board from FEN.
    
    Args:
        fen (str): FEN string representing the board position
        
    Returns:
        str: ASCII representation of the chess board
    """
    # Get the board part of FEN (everything before first space)
    board_fen = fen.split()[0]
    
    # Create empty board
    board = []
    
    # Convert FEN to 2D array
    for row in board_fen.split('/'):
        board_row = []
        for char in row:
            if char.isdigit():
                board_row.extend(['.'] * int(char))
            else:
                board_row.append(char)
        board.append(board_row)
    
    # Convert to string representation
    ranks = '87654321'
    files = 'abcdefgh'
    
    # Create board string with coordinates
    board_str = '    ' + '  '.join(files) + '\n'
    board_str += '   ' + '-' * 25 + '\n'
    
    for i, row in enumerate(board):
        board_str += f' {ranks[i]} | ' + ' '.join(row) + ' |\n'
    
    board_str += '   ' + '-' * 25 + '\n'
    board_str += '    ' + '  '.join(files)
    
    return board_str

def process_puzzle(puzzle_data):
    """Convert puzzle FEN into board representation and format the data"""
    board_repr = create_board_representation(puzzle_data['FEN'])
    
    # Extract first move as ground truth
    first_move = puzzle_data['Moves'].split()[0]

    return {
        "data_source": "chess_puzzles",
        "prompt": [{
            "role": "user",
            "content": create_puzzle_prompt(board_repr, puzzle_data)
        }],
        "ability": "chess",
        "reward_model": {
            "style": "rule",
            "ground_truth": first_move  # Now only using the first move
        },
        "extra_info": {
            "puzzle_id": puzzle_data['PuzzleId'],
            "rating": puzzle_data['Rating'],
            "rating_deviation": puzzle_data['RatingDeviation'],
            "popularity": puzzle_data['Popularity'],
            "nb_plays": puzzle_data['NbPlays'],
            "themes": puzzle_data['Themes'],
            "game_url": puzzle_data['GameUrl'],
            "opening_tags": puzzle_data['OpeningTags'],
            "fen": puzzle_data['FEN'],
            "moves": puzzle_data['Moves']
        }
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='../data/chess')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=None, 
                       help='Number of samples to process (None for all)')
    parser.add_argument('--dataset_path', default='lichess/chess-puzzles',
                       help='Path to the dataset')
    parser.add_argument('--output_format', choices=['parquet', 'json'], default='parquet',
                       help='Output format for the processed dataset')

    args = parser.parse_args()

    # Load the dataset
    dataset = datasets.load_dataset(args.dataset_path)
    full_dataset = dataset['train']  # Get the train split

    # Create train/test split
    test_size = min(1024, len(full_dataset) // 10)  # 10% or 1024, whichever is smaller
    splits = full_dataset.train_test_split(test_size=test_size, shuffle=True, seed=42)
    
    # Apply num_samples limit to training set if specified
    if args.num_samples is not None:
        splits['train'] = splits['train'].select(range(args.num_samples))

    # Process both splits
    processed_train = splits['train'].map(
        function=process_puzzle,
        remove_columns=splits['train'].column_names
    )
    
    processed_test = splits['test'].map(
        function=process_puzzle,
        remove_columns=splits['test'].column_names
    )

    # Save to local directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    if args.output_format == 'parquet':
        train_path = os.path.join(local_dir, 'puzzles_train.parquet')
        test_path = os.path.join(local_dir, 'puzzles_test.parquet')
        processed_train.to_parquet(train_path)
        processed_test.to_parquet(test_path)
    else:  # json
        train_path = os.path.join(local_dir, 'puzzles_train.json')
        test_path = os.path.join(local_dir, 'puzzles_test.json')
        processed_train.to_json(train_path)
        processed_test.to_json(test_path)

    print(f"Processed {len(processed_train)} training puzzles")
    print(f"Processed {len(processed_test)} test puzzles")
    print(f"Saved training data to {train_path}")
    print(f"Saved test data to {test_path}")

   

    