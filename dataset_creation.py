import pandas as pd
import os
from dotenv import load_dotenv
import yaml
from utils.common import read_yaml
from utils.llm import LargeLanguageModel


# Load environment variable
load_dotenv('./.env.development')

# Load model configuration
model_config_path = os.environ['model']
with open(model_config_path, 'r') as f:
    model_config = yaml.safe_load(f)

# Initialize LLM
llm = LargeLanguageModel(**model_config)

def llm_retriever(llm, questions, variation_number):
    variation_prompt = read_yaml(os.environ['prompt'])
    system_message = variation_prompt['system_message']
    prompt_template = variation_prompt['prompt']

    variations_data = pd.DataFrame()

    for index, question in enumerate(questions):
        prompt = prompt_template.format(query=question, variation_number=variation_number)

        # Generate variations for the current question
        variations = generate_variations(llm, prompt, system_message, variation_number)

        variations_data[f'Variations_{index}'] = variations

    return variations_data

def generate_variations(llm, prompt, system_message, number_variations):
    # Combine system message, prompt, and user query
    full_prompt = f"{system_message}\n{prompt}"

    # Generate variations for the combined prompt
    variations = llm.llm_runnable.invoke(query=full_prompt, number_variations=number_variations)

    return variations

def read_input_file(file_path):
    # Detect file type and read data accordingly
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.txt'):
        return pd.read_csv(file_path, header=None, names=['question'])
    else:
        raise ValueError("Unsupported file format. Please provide a TXT or CSV file.")

def main():
    # Load questions from TXT or CSV file
    input_file = "QuestionsGeneration.txt" 
    questions_data = read_input_file(input_file)

    # Extract questions from the DataFrame
    questions = questions_data['question'].tolist()

    variation_number = 10  # Adjust the number of variations to generate

    # Get variations
    variations_data = llm_retriever(llm, questions, variation_number)

    # Save variations to Excel file
    output_file = "generated_variations.xlsx"
    variations_data.to_excel(output_file, index=False)

    print(f"Variations saved to {output_file}")

if __name__ == "__main__":
    main()
