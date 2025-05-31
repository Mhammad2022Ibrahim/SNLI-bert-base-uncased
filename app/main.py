from fastapi import FastAPI
from .schemas import *
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tensorflow as tf


app = FastAPI(debug=True)

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("Mhammad2023/snli-bert-base-uncased",from_tf=True)
# model = AutoModelForSequenceClassification.from_pretrained("Mhammad2023/snli-bert-base-uncased")


@app.get("/")
async def root():
    return {"message": "Hello in our async SNLI BERT-base-uncased model API"}

@app.post("/predict", response_model=SNLIResponse)
async def predict(request: CreateSNLI):
    inputs = tokenizer(request.premise, request.hypothesis, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs).item()
        confidence = probs[0][predicted_class].item()
    # inputs = tokenizer(request.premise, request.hypothesis, return_tensors="tf")
    # outputs = model(**inputs)
    # probs = tf.nn.softmax(outputs.logits, axis=1)
    # predicted_class = tf.argmax(probs, axis=1).numpy()[0]
    # confidence = probs[0][predicted_class].numpy()
    
    return SNLIResponse(
        premise=request.premise,
        hypothesis=request.hypothesis,
        label=SNLIClass(predicted_class).name,
        confidence=confidence
    )