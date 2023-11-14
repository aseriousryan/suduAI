import uvicorn
from fastapi import FastAPI, File, UploadFile

from ingest import Ingest
from llm import SuduLLM

ingest_pdf = Ingest("chroma_db", "default_collection")
sudu_LLM = SuduLLM()

app = FastAPI()


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    with open('test.pdf', 'wb') as file:  
        file.write(content)
    response = ingest_pdf.run("test.pdf")
    return response
    
    
@app.post("/chat")
async def upload(msg, metadata):
    
    # ## Create the sudu llm
    sudu_LLM.create_sudu()

    # Infer
    response = sudu_LLM.infer_sudu({'query': msg}, 'collection_name')

    result = {
            'model_output': response['model_output'],
            'source_document': ''
        }

    return result

    

if __name__ == "__main__":
    uvicorn.run('main:app', host="0.0.0.0", port=8005, reload=True)
