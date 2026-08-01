# FastAPI

## What is it?

FastAPI is a Python web framework for building HTTP APIs — in the context of deploying AI systems, the standard way to expose a trained model as a service other systems can send requests to and get predictions back from.

[ML Workflow](../machine-learning/ml-workflow.md) ended its list of steps with "deployment: make the model available to produce predictions on live input." FastAPI is one of the most common ways that step is actually implemented in Python: wrapping a trained model behind an HTTP endpoint that accepts a request, runs the model, and returns a response.

## Why does it exist?

A trained model sitting in a saved file isn't usable by anything else until it's reachable over a network — a web application, a mobile app, or another internal service needs to send it input and get a prediction back, which requires a running service with a defined interface. FastAPI exists to make building that interface fast and safe: it uses Python type hints to automatically validate incoming request data and generate interactive API documentation, removing a large class of manual validation code and a common source of production bugs — a request with a malformed or missing field gets rejected automatically, with a clear error, before it ever reaches the model.

**When a synchronous request/response API fits, and when it doesn't:** wrapping a model behind a FastAPI endpoint fits when a caller needs an immediate response to a single, self-contained request — a classification result, an embedding, a single LLM completion. It's a poor fit for work that takes much longer than a typical request timeout, or for high-throughput batch scoring where millions of predictions get computed at once — those cases usually route through a message queue and an asynchronous worker process instead, since a client won't want to hold an open connection waiting minutes for a batch job to finish.

## How does it work?

- Define a **Pydantic model** — a typed class — describing the expected shape of a request's body. This is the type-hint-driven validation FastAPI is built around.
- Define an **endpoint** function, decorated with a route (e.g. `@app.post("/predict")`), that takes the validated request as an argument and returns a response.
- Load the model once, at startup, not per-request — a real production point covered below — and call it on the validated input inside the endpoint.
- FastAPI validates every incoming request against the Pydantic model automatically, rejecting anything that doesn't match with a clear error, and generates interactive API documentation from the same type definitions, with no extra code.

## Example

Wrapping the exact `LinearRegression` house-price model from [What is Machine Learning](../machine-learning/what-is-machine-learning.md) behind a FastAPI endpoint, and testing both a valid and an invalid request:

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

X = np.array([[50], [65], [80], [100], [120]])
y = np.array([150, 190, 230, 280, 330])
model = LinearRegression().fit(X, y)

app = FastAPI()

class HouseFeatures(BaseModel):
    size_sqm: float

class PricePrediction(BaseModel):
    predicted_price_thousands: float

@app.post("/predict", response_model=PricePrediction)
def predict(features: HouseFeatures):
    prediction = model.predict([[features.size_sqm]])
    return PricePrediction(predicted_price_thousands=float(prediction[0]))

client = TestClient(app)

valid = client.post("/predict", json={"size_sqm": 90})
print(valid.status_code, valid.json())

invalid = client.post("/predict", json={"size_sqm": "not a number"})
print(invalid.status_code, invalid.json())
```

```text
200 {'predicted_price_thousands': 253.97727272727272}
422 {'detail': [{'type': 'float_parsing', 'loc': ['body', 'size_sqm'], 'msg': 'Input should be a valid number, unable to parse string as a number', 'input': 'not a number'}]}
```

The valid request returns exactly the same predicted price (≈254) already verified in [What is Machine Learning](../machine-learning/what-is-machine-learning.md) — the model's behavior didn't change, only how it's reached. The invalid request — a string where a number was expected — never reaches the model at all: FastAPI rejects it automatically with a 422 status and a specific, structured error pointing at exactly which field failed and why, using nothing more than the type hint already written in `HouseFeatures`.

## Where is it used?

Serving a trained model or an LLM/RAG pipeline as an internal or external API, exposing a feature computation service, and generally any point where [ML Workflow](../machine-learning/ml-workflow.md)'s deployment step needs a concrete HTTP interface in Python.

## Advantages

- **Rejects malformed requests automatically**, based on type hints alone, before they ever reach the model.
- **Generates interactive API documentation from the same type definitions** used for validation, with no separate documentation step to maintain.
- **Handles the request/response plumbing**, so the actual model-serving code stays focused on the model itself.

## Limitations

- **A poor fit for long-running work or large batch jobs**, which don't suit a synchronous request/response shape at all.
- **Type-hint validation catches malformed requests, but says nothing about whether the model's output is actually correct** — that's a modeling and evaluation concern, not something the framework checks.
- **Doesn't manage the model's lifecycle on its own.** Loading, versioning, and updating the model in a running service is something the surrounding application still has to handle deliberately.

## Production considerations

- **Loading the model once at startup, not per-request, is essential.** Reloading a model on every incoming request adds latency and load that has nothing to do with the actual prediction.
- **The endpoint's response time is the model's inference time plus the framework's own overhead** — for an endpoint wrapping a large LLM call, that inference time, not FastAPI itself, is almost always the real bottleneck.
- **A model update needs a clear deployment story.** Replacing the model file underneath a running service without restarting it, or without careful coordination, can serve predictions from an inconsistent or partially-updated model.

## Common mistakes

- **Loading the model inside the endpoint function itself**, so it gets reloaded from disk on every single request instead of once at startup.
- **Assuming FastAPI's request validation is equivalent to validating the model's actual predictions.** It only checks that a request has the right shape, not that the resulting prediction is sensible.
- **Treating the API layer as the place to debug a bad prediction**, when a wrong output with a well-formed, valid request almost always points back to the model or its input data, not to FastAPI itself.

## Interview questions

### Basic

- What does FastAPI use to automatically validate incoming requests?
- Why does a malformed request never reach the model in the example above?

### Intermediate

- Why should a model be loaded once at startup rather than inside the endpoint function?
- When would a synchronous FastAPI endpoint be the wrong choice for serving a model?

### Advanced

- A FastAPI-served model's response time has grown significantly. How would you determine whether the framework or the model itself is the bottleneck?
- How would you update a model being served by a live FastAPI application without serving predictions from an inconsistent state during the transition?
