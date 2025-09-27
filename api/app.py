from fastapi import FastAPI
from pydantic import RootModel
from typing import Dict
from credit_eval import evaluate_credit

app = FastAPI(title="Credit Evaluation API")

class EntityRecord(RootModel[Dict]):
    """
    Accepts a dynamic dictionary of all entity fields.
    """
    pass

@app.post("/evaluate")
def evaluate(entity: EntityRecord):
    """
    Evaluate the credit of a single entity and return JSON feedback.
    """
    # entity.model_dump() converts RootModel to dict
    result = evaluate_credit(entity.model_dump())
    return {"feedback": result}
