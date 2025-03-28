import os
import pandas as pd
import os
import json 

TEST_QUESTIONS = [
    "Generate an SVG image of a cat riding a motorcycle",
    "Generate an SVG image of a dog walking upside down",
    "Generate an SVG image of a cat flying in the sky",
    "Generate an SVG image of a pink crocodile with a red collar",
    "Generate an SVG image of a blue elephant with three horizontal stripes",
    "Generate an SVG image of a green turtle with a patterned shell",
    "Generate an SVG image of a yellow submarine with two circular portholes",
    "Generate an SVG image of a red fire truck with two ladders",
    "Generate an SVG image of a white wizard wearing a pointed hat and a long beard",
    "Generate an SVG image of a brown horse with a saddle and bridle",
    "Generate an SVG image of a black cat with green eyes sitting on a white fence",
    "Generate an SVG image of a purple octopus with eight equal tentacles",
    "Generate an SVG image of a grey robot with two antennas and a digital display",
    "Generate an SVG image of a silver spaceship with two thrusters and a rounded nose",
    "Generate an SVG image of a gold coin with detailed circular engravings",
    "Generate an SVG image of a navy blue anchor with a curved design",
    "Generate an SVG image of a maroon flag with a white cross in the center",
    "Generate an SVG image of a turquoise parrot with a curved beak and two wings",
    "Generate an SVG image of an orange pumpkin with two carved eyes and a mouth",
    "Generate an SVG image of a lime green frog with two bulging eyes",
    "Generate an SVG image of a magenta flower with exactly five petals",
    "Generate an SVG image of a teal fish with a streamlined body and a dorsal fin",
    "Generate an SVG image of a beige book with a simple rectangular cover",
    "Generate an SVG image of a copper kettle with a curved spout and handle",
    "Generate an SVG image of a burgundy wine glass with a long stem",
    "Generate an SVG image of a violet guitar with six strings and a fretboard",
    "Generate an SVG image of a coral reef composed of basic coral shapes",
    "Generate an SVG image of a charcoal mountain with three distinct peaks",
    "Generate an SVG image of a mint green leaf with a smooth edge",
    "Generate an SVG image of a mustard yellow sun with evenly spaced rays",
    "Generate an SVG image of a cherry red apple with a single stem and leaf",
    "Generate an SVG image of a sky blue bird with a round body and outspread wings",
    "Generate an SVG image of a royal blue crown with three evenly spaced peaks",
    "Generate an SVG image of a forest green tree with a triangular canopy",
    "Generate an SVG image of a peach-colored cupcake with a swirl of frosting",
    "Generate an SVG image of a bronze bell with a clapper inside a circular frame",
    "Generate an SVG image of a cream-colored chair with four straight legs",
    "Generate an SVG image of a mint green ice cream cone with two uniform scoops",
    "Generate an SVG image of a pastel purple butterfly with two symmetrical wings",
    "Generate an SVG image of a deep red rose with a single bloom and defined petals",
    "Generate an SVG image of a cobalt blue boat with a single mast and curved hull",
    "Generate an SVG image of a clear glass bottle with a simple label outline",
    "Generate an SVG image of a dark green cactus with two arms of equal length",
    "Generate an SVG image of a light blue cloud with a smooth, rounded outline",
    "Generate an SVG image of a brown tree trunk with a uniform bark texture",
    "Generate an SVG image of a silver key with a detailed bow shape",
    "Generate an SVG image of a gold crown with five small, evenly spaced jewels",
    "Generate an SVG image of a red barn with a pitched roof and two identical doors",
    "Generate an SVG image of a blue fence with evenly spaced vertical posts",
    "Generate an SVG image of a green field with rows of identical crop icons",
    "Generate an SVG image of a white house with one central window and door",
    "Generate an SVG image of a black car with four equal wheels and a boxy design",
    "Generate an SVG image of a blue bicycle with two circular wheels and a straight frame",
    "Generate an SVG image of a yellow school bus with rectangular windows",
    "Generate an SVG image of a red airplane with a single tail and straight wings",
    "Generate an SVG image of a brown train with three connected carriages",
    "Generate an SVG image of a gray helicopter with one main rotor",
    "Generate an SVG image of a green truck with a rectangular cargo box",
    "Generate an SVG image of a navy blue submarine with a vertical periscope",
    "Generate an SVG image of a black motorcycle with two circular wheels",
    "Generate an SVG image of a white ship with a curved hull and a single sail",
    "Generate an SVG image of a red lighthouse with a circular light on top",
    "Generate an SVG image of a blue tent with a rectangular shape and a peaked roof",
    "Generate an SVG image of a green backpack with two parallel straps",
    "Generate an SVG image of a purple hat with a wide, flat brim",
    "Generate an SVG image of a black boot with a defined sole and lacing",
    "Generate an SVG image of a brown wallet with a simple, rectangular design",
    "Generate an SVG image of a gray clock with two hands and numeric markers",
    "Generate an SVG image of a gold watch with a round face and minimal markings",
    "Generate an SVG image of a silver smartphone with a rectangular screen",
    "Generate an SVG image of a blue laptop with a hinged display",
    "Generate an SVG image of a black camera with a prominent lens and viewfinder",
    "Generate an SVG image of a white tablet with a single circular home button",
    "Generate an SVG image of a red smartphone with a simple icon layout",
    "Generate an SVG image of a green calculator with numeric buttons arranged in a grid",
    "Generate an SVG image of a blue printer with a visible paper tray",
    "Generate an SVG image of a gray desktop computer with a flat monitor",
    "Generate an SVG image of a black gaming console with a single controller",
    "Generate an SVG image of a white refrigerator with a digital temperature display",
    "Generate an SVG image of a silver microwave with a round dial",
    "Generate an SVG image of a red toaster with two parallel slots",
    "Generate an SVG image of a green blender with a transparent container",
    "Generate an SVG image of a blue coffee maker with a drip system",
    "Generate an SVG image of a brown sofa with three uniform cushions",
    "Generate an SVG image of a gray table with four straight legs",
    "Generate an SVG image of a white chair with a simple backrest and seat",
    "Generate an SVG image of a black bed with a headboard and footboard of equal width",
    "Generate an SVG image of a red lamp with a circular shade",
    "Generate an SVG image of a blue wall clock with clearly marked numbers",
    "Generate an SVG image of a green door with a rectangular window pane",
    "Generate an SVG image of a yellow mailbox with a flag on its side",
    "Generate an SVG image of a purple pen with a standard click mechanism",
    "Generate an SVG image of a black pencil with a small eraser on top",
    "Generate an SVG image of a white book with a visible spine and cover",
    "Generate an SVG image of a red backpack with two evenly spaced zippers",
    "Generate an SVG image of a blue water bottle with a screw cap",
    "Generate an SVG image of a green umbrella with a curved handle",
    "Generate an SVG image of a brown pair of shoes with laces and a flat sole",
    "Generate an SVG image of a gray hat with a flat, wide brim",
    "Generate an SVG image of a black tie with a narrow design",
    "Generate an SVG image of a white t-shirt with a round neckline",
    "Generate an SVG image of a red scarf with parallel stripes",
    "Generate an SVG image of a blue jacket with a zipper and two side pockets",
    "Generate an SVG image of a pink wizard with a pointed hat and a long beard",
    "Generate an SVG image of a purple knight in armor holding a sword and shield",
    "Generate an SVG image of a green robot with a boxy design and simple facial features"
]

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

if __name__ == '__main__':
    # Directory containing the variant result JSON files
    variants_dir = '/home/ubuntu/o1-replication-central/CustomTinyZero/data/svg_variants'

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
                
                # Process each root node in the tree
                for root_node in variants_data:
                    extract_variants(root_node)
                

    variants_to_parquet(TRAIN_QUESTIONS,TEST_QUESTIONS, '/home/ubuntu/o1-replication-central/CustomTinyZero/data/svg_variants')