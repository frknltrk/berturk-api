# https://fastapi.tiangolo.com/tutorial/first-steps/
# https://fastapi.tiangolo.com/tutorial/path-params/
# https://www.sbert.net/

import uvicorn
from fastapi import FastAPI, Depends
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("dbmdz/bert-base-turkish-uncased")
app = FastAPI()

@app.get("/{text}")
def encode(text: str):
  return model.encode(text)

if __name__ == "__main__":
  uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)
