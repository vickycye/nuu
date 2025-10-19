from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import upload, status
from app.core.config import settings 

app = FastAPI(title="Nuu", version="0.1.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"], # configure later properly for production
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# include routers
app.include_router(upload.router, prefix="/api", tags=['upload'])
app.include_router(status.router, prefix="/api", tags=['status'])

@app.get("/")
async def root():
    return {"message": "Nuu 3D Scanner API"}

if __name__ == "__main__":
    import uvicorn # this is a lightweight, fast ASGI server for python. 
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
