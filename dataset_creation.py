import pandas as pd
import os
import yaml

from dotenv import load_dotenv
from utils.common import read_yaml
from utils.llm import LargeLanguageModel

def llm_retriever(llm, questions, variation_number):
    variation_prompt = read_yaml('./prompts/generate_variation_prompt.yml')
    system_message = variation_prompt['system_message']
    prompt_template = variation_prompt['prompt']

    variations_data = []

    for index, question in enumerate(questions):
        prompt = prompt_template.format(query=question, number_variations=variation_number)

        # Generate variations for the current question
        generated_variations = generate_variations(llm, prompt, system_message)
        generated_variations = generated_variations.split('```\n')[1].split('\n```')[0]
        generated_variations = generated_variations.split('\n')
        
        # add the original question to the top of the list as the first variation
        generated_variations = [question] + generated_variations

        # Store the entire variation sentence in a separate cell
        variations_data.append({
            'question': question,
            'variations': generated_variations
        })


    return variations_data


def generate_variations(llm, prompt, system_message):
    generated_variations = llm.llm_runnable.invoke({'system_message': system_message, 'prompt': prompt})
    return generated_variations

def read_input_file(file_path):
    # Detect file type and read data accordingly
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            questions = f.readlines()
        return pd.DataFrame({'question': questions})
    else:
        raise ValueError("Unsupported file format.")

def main():
    # Load environment variable
    load_dotenv('./.env.development')

    # Load model config
    model_config_path = os.environ['model']
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    # Initialize LLM
    llm = LargeLanguageModel(**model_config)
    
    # Load questions from TXT or CSV file
    input_file = "test.txt" 
    questions_data = read_input_file(input_file)

    # Extract questions
    questions = questions_data['question'].tolist()

    variation_number = 2  # Adjust the number of variations to generate

    # Save variations
    output_file = "test.xlsx"
    writer = pd.ExcelWriter(output_file)
    variations_data = llm_retriever(llm, questions, variation_number)
    for idx, data in enumerate(variations_data):
        question = data['question']
        variations = pd.DataFrame(data['variations'], columns=['variation'])
        variations.to_excel(writer, sheet_name=f'{idx}', index=False)

    writer.close()
    print(f"Variations saved to {output_file}")

if __name__ == "__main__":
    main()
