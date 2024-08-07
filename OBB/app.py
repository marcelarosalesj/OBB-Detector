from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def hello_world():
    import sys
    print(f"SYSTEM => {sys.path}")
    from .OBB import obb_2d_track
    print(f"finished import")
    return "<p>Hello, World!</p>"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
