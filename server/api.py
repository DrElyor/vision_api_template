import base64
import multiprocessing as mp
import os
import time
import uuid
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="VISION API TEMPLATE",
        version="0.1.0",
        description="OpenAPI Schema for Vision App",
        routes=app.routes,
    )
    # openapi_schema["info"]["x-logo"] = {
    #     "url": "" # SET LOGO URL
    # }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app = FastAPI()
app.openapi = custom_openapi

origins = [
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def query_to_image(contents):
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.post("/api/demo")
async def api_demo(file: UploadFile = File(...)):
    contents = await file.read()
    img=query_to_image(contents)
    img_dimensions = str(img.shape)
    
    ### Define process 
    img_out=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    ###
    
    _, encoded_img = cv2.imencode(".jpg", img_out)
    encoded_img = base64.b64encode(encoded_img)
    encoded_url = "data:image/jpg;base64," + str(encoded_img)[2:-1]
    return {
        "filename": file.filename,
        "dimensions": img_dimensions,
        "encoded_img": encoded_url,
    }


@app.get("/alive")
def alive():
    return {"status": "IAMALIVE"}
