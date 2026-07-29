from fastapi import FastAPI

app = FastAPI(
    title="Notes Assistant API",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "Welcome to Notes Assistant"}