from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os

from src.routers import predict, chat, alerts, intelligence

app = FastAPI(title="FloodGuard Pakistan V3 API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Static and Templates
base_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(base_dir, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

# Include Routers
app.include_router(predict.router, prefix="/api/predict", tags=["prediction"])
app.include_router(chat.router, prefix="/api/chat", tags=["chatbot"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["alerts"])
app.include_router(intelligence.router, prefix="/api/data", tags=["data_intelligence"])

@app.get("/")
async def serve_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/mock")
async def serve_mock_dashboard(request: Request):
    return templates.TemplateResponse("mock.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
