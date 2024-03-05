import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import os
from tqdm import tqdm 


# Load environment variables
load_dotenv('./.env.development')

client = OpenAI(api_key='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',)

# Function to evaluate answers using OpenAI's model
def evaluate_answer(correct_answer, candidate_answer):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Rate the correctness of the candidate answer based on the correct answer on a scale from 0 to 1. (All answer in numerics, no any english word in your answer!)"},
                {"role": "user", "content": f"Correct Answer: {correct_answer}"},
                {"role": "user", "content": f"Candidate Answer: {candidate_answer}"}
            ],
        )
        # Parse the response correctly based on the API's structure
        score = response.choices[0].message.content
    except KeyError:
        # Handle cases where the expected keys are not in the response
        score = "Error parsing response"
    except Exception as e:
        # Handle any other exceptions
        score = f"Unexpected error: {str(e)}"
    return score

# Ensure you've correctly pointed to your CSV file
# df = pd.read_csv('test.csv')
df = pd.read_excel('cot_openchat_1.xlsx')

# Evaluate similarity scores and add them to the DataFrame
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating"):
    correct_answer = row['ground_truth']
    candidate_answer = row['llm']
    df.loc[index, 'Similarity_Score'] = evaluate_answer(correct_answer, candidate_answer)
print(df)

df['New_Similarity_Score'] = pd.to_numeric(df['Similarity_Score'], errors='coerce')
df['GPT rating'] = df['New_Similarity_Score'].apply(lambda x: 1 if x >= 0.5 else 0)

print(df)
df.to_excel('evaluation_cot_openchat_1.xlsx', index=False)

