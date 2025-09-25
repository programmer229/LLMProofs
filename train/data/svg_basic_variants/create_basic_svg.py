from svg_basic import TRAIN_QUESTIONS, TEST_QUESTIONS
import os

def variants_to_parquet(train_questions, test_questions, output_path: str) -> None:
    samples = []
    test_samples = []
    # Define an instruction for the incorrect questions.
    instruction_following = "<instruction> Generate an SVG image of the following description. Reason about how to generate the SVG beforehand, then write the final SVG in <svg_image> </svg_image> tags. </instruction>"

    # Loop over each question.
    for idx, train_question in enumerate(train_questions):
        # Build the prompt by combining the question with the instruction.
        prompt_content = f"{train_question}\n{instruction_following}"
        
        # Build a sample dictionary
        sample = {
            "data_source": "llm_judge_svg",
            "prompt": [{
                "role": "user",
                "content": prompt_content
            }],
            "ability": "integration",
            "reward_model": {
                "style": "rule",
                "ground_truth": train_question
            },
            "extra_info": {
                "question_index": idx,
                "question_id": train_question
            }
        }
        samples.append(sample)

    # Create test samples using the base question
    for idx, test_question in enumerate(test_questions):
        prompt_content = f"{test_question}\n{instruction_following}"
        test_sample = {
            "data_source": "llm_judge_svg",
            "prompt": [{
                "role": "user", 
                "content": prompt_content
            }],
            "ability": "integration",
            "reward_model": {
                "style": "rule",
                "ground_truth": test_question
            },
            "extra_info": {
                "question_index": idx,
                "question_id": test_question
            }
        }
        test_samples.append(test_sample)
    
    # Define a local output directory and ensure it exists.
    os.makedirs(output_path, exist_ok=True)

    # Save the samples to JSON files
    import json
    with open(os.path.join(output_path, f'train.json'), 'w') as f:
        json.dump(samples, f, indent=4)
    with open(os.path.join(output_path, f'test.json'), 'w') as f:
        json.dump(test_samples, f, indent=4)
    
    # Save the samples to Parquet files
    import pandas as pd
    df = pd.DataFrame(samples)
    df.to_parquet(os.path.join(output_path, f'train.parquet'))
    
    test_df = pd.DataFrame(test_samples)
    test_df.to_parquet(os.path.join(output_path, f'test.parquet'))
    
    print(f"Train samples saved to {output_path}/train.parquet")
    print(f"Test samples saved to {output_path}/test.parquet")

if __name__ == "__main__":
    variants_to_parquet(TRAIN_QUESTIONS, TEST_QUESTIONS, "/home/ubuntu/o1-replication-central/CustomTinyZero/data/svg_basic_variants")