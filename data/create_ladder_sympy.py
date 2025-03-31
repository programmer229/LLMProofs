import os
import pandas as pd
import os
import json 

TEST_QUESTIONS = [
    "integrate(x + (x**0.5)/(1 + (x**0.5)), x)",
    
    "integrate(exp(x + 1)/(exp(x) + 1), x)",
    
    "integrate((3*sin(x) - sin(3*x))**(1/3), x)",
    
    "integrate(log(x**(log(x**x)))/x**2, x",
    
    "integrate(cos(20*x)*sin(25*x), x, -pi/2, pi/2)",
    
    "integrate(sin(x)*cos(x)*tan(x)*cot(x)*sec(x)*csc(x), x, 0, 2*pi)",
    
    "integrate((x*log(x)*cos(x) - sin(x))/(x*log(x)**2), x)",
    
    "integrate((2*x - 1 + log(2*x)), x, 1, 2)",
    
    "integrate(x**2024*(1 - x**2025)**2025, x, 0, 1)",
    
    "integrate((x - 1/2)*(x - 1)*x, x, 0, 10)",
    
    "integrate(floor(x)/2, x, 0, 20)",
    
    "integrate((exp(2*x)*(x**2 + x))/(x*exp(x)*4 + 1), x)",
    
    "integrate(sec(x)**4 - tan(x)**4, x)",
    
    "integrate(sqrt(x*(1 - x)), x, 0, 1)",
    
    "integrate(sin(4*x)*cos(x)/(cos(2*x)*sin(x)), x)",
    
    "integrate(sin(x)*sinh(x), x)",
    
    "integrate(sin(x)*cos(pi/3 - x), x, 0, pi/3)",
    
    "integrate((cos(x) + cos(x + 2*pi/3) + cos(x - 2*pi/3))**2, x)",
    
    "integrate(sum((-1)**k*x**(2*k)), x, 0, 1)",
    # Add more integrals as needed
]


def variants_to_parquet(train_questions, test_questions, output_path: str) -> None:
    samples = []
    test_samples = []
    # Define an instruction for the incorrect questions.
    instruction_following = "<instruction> Solve the following integral. Provide ONLY your antiderivative as a valid Python sympy expression e.g  <answer>cos(x**2)+ ln(x)+(1/3)*x**3</answer> wrapped in a <answer> tags. Importantly, put * between terms you want to multiply! Show your full working out before solving, don't include any constants of integration. DO NOT OUTPUT IN LATEX FORMAT. OUTPUT IN SYMPY in <answer> tags. </instruction>"

    # Loop over each question.
    for idx, train_question in enumerate(train_questions):
        # Build the prompt by combining the question with the instruction.
        prompt_content = f"{train_question}\n{instruction_following}"
        
        # Build a sample dictionary
        sample = {
            "data_source": "llm_judge_integration_sympy",
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
            "data_source": "integration_numeric",
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

if __name__ == '__main__':
    # Directory containing the variant result JSON files
    variants_dir = '/home/ubuntu/o1-replication-usmid/CustomTinyZero/data/ladder_variants'

    # List to store all variants
    TRAIN_QUESTIONS = []

    # Iterate through all JSON files in the directory
    for filename in os.listdir(variants_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(variants_dir, filename)
            
            # Read and parse each JSON file
            with open(file_path, 'r') as f:
                variants_data = json.load(f)
                
                def extract_variants(node):
                    # Extract variants from current node
                    variants = node.get('variants', [])
                    TRAIN_QUESTIONS.extend(variants)
                    
                    # Recursively process children
                    for child in node.get('children', []):
                        extract_variants(child)
                
                # Process the tree starting from root
                for root_node in variants_data['tree']:
                    extract_variants(root_node)
                

    variants_to_parquet(TRAIN_QUESTIONS, TEST_QUESTIONS, '/home/ubuntu/o1-replication-usmid/CustomTinyZero/data/ladder_sympy')