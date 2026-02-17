from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok", "service": "gotti-api"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Example endpoint to list strategies
@app.get("/strategies")
def list_strategies():
    strategies_dir = "strategies"
    strategies = []
    if os.path.exists(strategies_dir):
        for f in os.listdir(strategies_dir):
            if f.endswith(".py") and f != "__init__.py":
                strategies.append(f)
    return {"strategies": strategies}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
