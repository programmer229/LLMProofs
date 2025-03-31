import streamlit as st
import json
import base64

"""
Instructions:
1. Set PATH_TO_LOGS to the path of the question_logs.json file for your SVG experiments.
2. Run this file with the command: streamlit run svg_viewer.py
"""

PATH_TO_LOGS = "/home/ubuntu/o1-replication-central/CustomTinyZero/checkpoints/svg_judge_experiments/qwen2.5_7b_svg_gpt4o_mini/question_logs.json"

#################################################################################################

# Load the SVGs from the JSON file
with open(PATH_TO_LOGS, 'r') as f:
    svg_data = json.load(f)

# Create radio button to select display mode
display_mode = st.radio("Select display mode", ["1 SVG", "4 SVGs", "9 SVGs"])

# Create dropdown selectors
col1, col2 = st.columns(2)
with col1:
    batch_idx = st.selectbox("Select batch", range(len(svg_data)))

with col2:
    if batch_idx < len(svg_data):
        batch = svg_data[batch_idx]
        if display_mode == "1 SVG":
            question_ids = list(batch.keys())
            question_id = st.selectbox("Select question", question_ids)
        else:
            num_svgs = {"4 SVGs": 4, "9 SVGs": 9}[display_mode]
            max_range = len(batch.keys())
            ranges = [f"{i}-{min(i+num_svgs-1, max_range-1)}" for i in range(0, max_range, num_svgs)]
            selected_range = st.selectbox("Select range", ranges)
            start_idx = int(selected_range.split('-')[0])

# Display single SVG mode
if display_mode == "1 SVG" and batch_idx < len(svg_data):
    batch = svg_data[batch_idx]
    if question_id in batch:
        svg_details = batch[question_id]
        svg_text = svg_details["extracted_solution"]
        if svg_text:
            st.code(svg_text, language="xml")
            st.text(f"Judge Score: {svg_details['total_score']}")
            if svg_details["base64_solution"] != -1:
                st.image(f"data:image/png;base64,{svg_details['base64_solution']}")
                st.text(f"Prompt: {svg_details['model_solution'].split('<instruction>')[0].strip()}")
        else:
            st.error("No valid SVG found")

# Display grid of SVGs
# Display grid of SVGs
# Display grid of SVGs
else:
    num_svgs = {"4 SVGs": 4, "9 SVGs": 9}[display_mode]
    if display_mode == "4 SVGs":
        cols_per_row = 2
        rows_per_grid = 2
        font_size = "12px"  # Slightly larger font for 4-grid view
    else:  # "9 SVGs"
        cols_per_row = 3
        rows_per_grid = 3
        font_size = "10px"  # Smaller font for 9-grid view
    
    batch = svg_data[batch_idx]
    question_ids = list(batch.keys())[start_idx:start_idx + num_svgs]
    
    # Calculate grid cell size based on display mode
    grid_cell_size = 350 if display_mode == "4 SVGs" else 250  # Increased sizes
    
    # Create a container with fixed width to enforce grid layout
    grid_container = st.container()
    with grid_container:
        for row in range(rows_per_grid):
            cols = st.columns(cols_per_row)
            for col in range(cols_per_row):
                idx = row * cols_per_row + col
                if idx < len(question_ids):
                    question_id = question_ids[idx]
                    with cols[col]:
                        svg_details = batch[question_id]
                        
                        # Calculate the max dimensions to fit inside the container with padding
                        max_dim = grid_cell_size - 40  # Account for padding and border
                        
                        # Get a truncated prompt to ensure it fits
                        prompt_text = svg_details['model_solution'].split('<instruction>')[0].strip()
                        max_prompt_length = 100 if display_mode == "4 SVGs" else 80
                        truncated_prompt = prompt_text[:max_prompt_length] + ("..." if len(prompt_text) > max_prompt_length else "")
                        
                        # Create a single HTML block with both the container and the image inside it
                        if svg_details["base64_solution"] != -1:
                            st.markdown(f"""
                                <div style="width: {grid_cell_size}px; height: {grid_cell_size + 80}px; 
                                          display: flex; flex-direction: column; align-items: center; 
                                          justify-content: flex-start; margin: 5px; padding: 10px; 
                                          border: 3px solid black; border-radius: 5px; overflow: hidden;">
                                    <div style="width: 100%; height: {grid_cell_size - 40}px; display: flex; align-items: center; justify-content: center;">
                                        <img src="data:image/png;base64,{svg_details['base64_solution']}" 
                                             style="max-width: {max_dim}px; max-height: {max_dim}px; object-fit: contain;">
                                    </div>
                                    <div style="text-align: center; margin-top: 5px; width: 100%; font-size: {font_size}; overflow: hidden;">
                                        <strong>Judge Score:</strong> {svg_details['total_score']}<br>
                                        <strong>Prompt:</strong> {truncated_prompt}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div style="width: {grid_cell_size}px; height: {grid_cell_size + 80}px; 
                                          display: flex; flex-direction: column; align-items: center; 
                                          justify-content: flex-start; margin: 5px; padding: 10px; 
                                          border: 3px solid black; border-radius: 5px; overflow: hidden;">
                                    <div style="width: 100%; height: {grid_cell_size - 40}px; display: flex; align-items: center; justify-content: center;">
                                        <p style="color: red; text-align: center;">SVG invalid for this question.</p>
                                    </div>
                                    <div style="text-align: center; margin-top: 5px; width: 100%; font-size: {font_size}; overflow: hidden;">
                                        <strong>Judge Score:</strong> {svg_details['total_score']}<br>
                                        <strong>Prompt:</strong> {truncated_prompt}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)