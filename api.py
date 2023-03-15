from fastapi import FastAPI
from predictor import Predictor
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from util import load_model_config

checkpoint= './saved_models/chatbot.pt'
tokenizer = './tokenizer/tokenizer.pkl'
limit_token =  64



origins = [
    "http://localhost:4200",
    "http://localhost:3000"
]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


predictor = Predictor(checkpoint=checkpoint, limit_token=limit_token, tokenizer=tokenizer)

class Input(BaseModel):
    input: str

@app.post('/chat')
def translate(data:Input):
    return {"result": predictor.predict(data.input)}