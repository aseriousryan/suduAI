import uvicorn
from fastapi import FastAPI, File, UploadFile

from ingest import Ingest

ingest_pdf = Ingest("chroma_db", "default_collection")
app = FastAPI()


@app.post("/upload")
async def upload(metadata, file: UploadFile = File(...)):
    content = await file.read()
    with open('test.pdf', 'wb') as file:  
        file.write(content)
    response = ingest_pdf.run("test.pdf", metadata)
    return response
    
    
@app.post("/chat")
async def upload(msg):
    return msg

if __name__ == "__main__":
    uvicorn.run('main:app', host="127.0.0.1", port=8005, reload=True)