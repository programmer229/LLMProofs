import os
import pandas as pd
import os
import json

TEST_QUESTIONS = [
    'integrate(12*sinh(x)**2, x)',
    'integrate(cosh(x)**2, x)',
    'integrate(x/(1+x**2), x)',
    'integrate(exp(-x)*cos(x), x)',
    'integrate(exp(2*x)*sin(3*x), x)',
    'integrate(x*log(x**2+1), x)',
    'integrate((exp(x)-1)/(exp(x)+2), x)',
    'integrate(x**2*cos(x), x)',
    'integrate((x**3+1)/(x**2+1), x)',
    'integrate(x/(x**4+1), x)',
    'integrate(exp(x)*sqrt(1+exp(2*x)), x)',
    'integrate(x**3/(x**2+1)**2, x)',
    'integrate(sinh(2*x), x)',
    'integrate(cosh(2*x), x)',
    'integrate(sin(3*x)*cos(2*x), x)',
    'integrate((x+2)/(x**2+2*x+2)+x^3, x)',
    'integrate((x-2)/(x**2-2*x+2), x)',
    'integrate(1/sqrt(x**2+9), x)',
    'integrate(x/(x**2+9), x)',
    'integrate(32*x*sin(x)**2, x)',
    'integrate(x*cos(x)**2, x)',
    'integrate(sin(x)*cos(x)**2, x)',
    'integrate(x*sin(x)*cos(x), x)',
    'integrate((x**3+2*x)/(x**2+1), x)',
    'integrate(x**4/(x**2+1), x)',
    'integrate((x**2-1)/(x**2+1), x)',
    'integrate(x**3*exp(-x), x)',
    'integrate(sin(x)/(1+cos(x)**2), x)',
    'integrate((x+1)**2/(x**2+1), x)',
    'integrate(x*sin(2*x)*cos(3*x), x)',
    'integrate(x*sin(x**2), x)',
    'integrate(x*cos(x**2), x)',
    'integrate(12*exp(-x)*cosh(x), x)',
    'integrate(x/(1+x**4), x)',
    'integrate(x**3/(1+x**4), x)',
    'integrate((x+2)/(x**2+4), x)',
    'integrate((x-2)/(x**2+4), x)',
    'integrate(x*atan(x), x)',
    'integrate(x*(1+x**2)**(-3/2), x)',
    'integrate(x*exp(-x**2), x)',
    'integrate(atan(x)/(1+x**2), x)',
    'integrate(x**2*exp(x)*sin(x), x)', 
    'integrate((x**2+1)/(x**3+x), x)',
    'integrate(acosh(x)/x, x)',
    'integrate(log(x**2+1)/(x**2+1), x)',
    'integrate(atan(x**2), x)',
    'integrate((x+1)/sqrt(x**2+1), x)',
    'integrate((x-1)/sqrt(x**2+1), x)',
    'integrate(sin(x)**2*cos(x)+x^12, x)',
    'integrate(x/sqrt(1+x**2), x)',
    'integrate(exp(-2*x)*sin(x), x)',
    'integrate(exp(-x)*sin(2*x), x)',
    'integrate(exp(-x)*cos(2*x), x)',
    'integrate(sin(x)*cos(2*x), x)',
    'integrate((2*x+1)/(x**2+x+1), x)',
    'integrate(acos(x)/sqrt(1-x**2), x)',
    'integrate(x*sqrt(x**2+4), x)',
    'integrate(x/(sqrt(x**2+4)), x)',
    'integrate((x+1)/(x**2+3), x)',
    'integrate((x-1)/(x**2+3), x)',
    'integrate(x/(x**4+4), x)',
    'integrate(1/(x**2+4*x+5), x)',
    'integrate((2*x+3)/(x**2+4*x+5), x)',
    'integrate(x**3/(x**4+1), x)',
    'integrate(1023*sin(x)**3, x)',
    'integrate(atan(x), x)',
    'integrate(exp(x)*sinh(x), x)',
    'integrate(exp(x)*cosh(x), x)',
    'integrate(sinh(x)**2+x^3, x)',
    'integrate(1/cosh(x)**2, x)',
    'integrate(exp(-x)*cosh(x), x)',
    'integrate(x**3*atan(x), x)',
    'integrate(x/(1+x**2)**2, x)',
    'integrate(exp(x)/sqrt(1+exp(2*x)), x)',
    'integrate((x+2)/(x**2+2*x+2), x)',
    'integrate(exp(-x**2), x)',
    'integrate(sin(x)**2*cos(x), x)',
    'integrate(x*sin(x)**2, x)',
    'integrate(5*x*cos(x)**2, x)',
    'integrate(cos(x) + asinh(x), x)',
    'integrate(sin(x)**2/(1+cos(x)), x)',
    'integrate(7*sinh(x)**2, x)',
    'integrate(5*x/(1+x**2), x)',
    'integrate(8*exp(2*x)*sin(3*x), x)', 
    'integrate(3*x**2*cos(x), x)',
    'integrate(6*x/(x**4+1), x)',
    'integrate(11*x**3/(x**2+1)**2, x)',
    'integrate(x^3 + 4*cosh(2*x), x)',
    'integrate(x + 7*(x+2)/(x**2+2*x+2), x)',
    'integrate(sin(x) + 3*(x-2)/(x**2-2*x+2), x)',
    'integrate(cos(x) + 13*x*sin(x)**2, x)',
    'integrate(sin(x)cost(x) + 2*x*cos(x)**2, x)',
    'integrate(7*x*sin(x)*cos(x), x)',
    'integrate(9*x**4/(x**2+1), x)',
    'integrate(5*x**3*exp(-x), x)',
    'integrate(4*(x+1)**2/(x**2+1), x)',
    'integrate(8*exp(-x)*cosh(x), x)',
    'integrate(4*x**3/(1+x**4), x)',
    'integrate(6*(x-2)/(x**2+4), x)',
    'integrate(7*x*(1+x**2)**(-3/2), x)'
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
            "data_source": "llm_judge_integration",
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
    variants_dir = '/home/ubuntu/o1-replication-usmid/CustomTinyZero/data/LlamaVariants/variant_results_10Q'

    # List to store all variants
    TRAIN_QUESTIONS = []

    # Iterate through all JSON files in the directory
    for filename in os.listdir(variants_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(variants_dir, filename)
            
            # Read and parse each JSON file
            with open(file_path, 'r') as f:
                variants_data = json.load(f)
                
                # Extract variants from each dictionary in the file
                for variant_dict in variants_data:
                    if 'variant' in variant_dict:
                        TRAIN_QUESTIONS.append(variant_dict['variant'])

    variants_to_parquet(TRAIN_QUESTIONS,TEST_QUESTIONS, '/home/ubuntu/o1-replication-usmid/CustomTinyZero/data/integration_3b_llmjudge')