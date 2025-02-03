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
    parser.add_argument('--local_dir', default='./lichess_puzzles')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=None, 
                       help='Number of samples to process (None for all)')
    parser.add_argument('--dataset_path', default='lichess/chess-puzzles',
                       help='Path to the dataset')
    parser.add_argument('--output_format', choices=['parquet', 'json'], default='json',
                       help='Output format for the processed dataset')
    parser.add_argument('--min_rating', type=int, default=0,
                       help='Minimum puzzle rating to include (default: 0)')
    parser.add_argument('--max_rating', type=int, default=500,
                       help='Maximum puzzle rating to include (default: 3000)')

    args = parser.parse_args()

    # Load the dataset
    dataset = datasets.load_dataset(args.dataset_path)

    # Select subset if num_samples is specified
    if args.num_samples is not None:
        dataset = dataset['train'].select(range(args.num_samples))
    else:
        dataset = dataset['train']  # Get the train split

    # Filter by rating range
    dataset = dataset.filter(lambda x: args.min_rating <= x['Rating'] <= args.max_rating)

    # Process the dataset
    processed_dataset = dataset.map(
        function=process_puzzle,
        remove_columns=dataset.column_names
    )

    # Save to local directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    if args.output_format == 'parquet':
        output_path = os.path.join(local_dir, 'puzzles.parquet')
        processed_dataset.to_parquet(output_path)
    else:  # json
        output_path = os.path.join(local_dir, 'puzzles.json')
        processed_dataset.to_json(output_path)

    print(f"Processed {len(processed_dataset)} puzzles")
    print(f"Saved to {output_path}")

   

    