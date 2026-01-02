from fastapi import FastAPI
from src.controller.service_controller import inspection_router


app = FastAPI()
app.include_router(inspection_router, tags=['DISKS INSPECTION'], prefix="/ai")
