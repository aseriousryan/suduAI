from sentence_transformers import SentenceTransformer
from utils.common import tokenize
import os
from dotenv import load_dotenv
from utils.common import ENV
load_dotenv(f'./.env.{ENV}')

emb_model = SentenceTransformer(os.environ['prompt_example_retriever_sentence_transformer'])

def store_question_embedding(question_text):
    # Convert the question into an embedding value
    emb = emb_model.encode(question_text, convert_to_numpy=True).tolist()

    # Prepare the question description
    question_info = question_text
    ques_token_length = len(tokenize(os.environ['tokenizer'], question_info))

    # Return the relevant variables
    return question_info, ques_token_length, emb
