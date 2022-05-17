from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import service
import base64
from fastapi.responses import FileResponse
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/hello")
async def pong():
    return {"message": "hello everyone"}

@app.post("/caption/get")
def caption(file: str):
    imgstr = file
    imgdata = base64.b64decode(imgstr)
    filename = 'getcaption.jpg'
    with open(filename, 'wb') as f:
        f.write(imgdata)
    caption = service.get_caption(filename)
    return {"caption": caption}

class SayCaptionModel(BaseModel):
    caption: str

@app.post("/caption/audio")
def predict_caption(caption: str):
    print(caption)
    file_path = service.get_caption_filepath(caption)
    return {"file_path": file_path}

@app.get("/caption/sayit/{file_path}")
def say_caption(file_path: str):
    return FileResponse(path=file_path, filename=file_path, media_type='application/octet-stream')