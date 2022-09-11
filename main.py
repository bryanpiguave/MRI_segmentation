import uvicorn
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, Response 
from starlette.responses import StreamingResponse
import cv2
import numpy as np
from tensorflow.keras.models import Model, load_model, save_model
app = FastAPI(debug=True)

#model = load_model("Unet.h5", compile=False)

def model(img):
    return img

def apply_mask(img:np.array,mask:np.array):
    mask = mask>0
    color = np.array([0,255,0], dtype='uint8')
    masked_img = np.where(mask[...,None], color, img)
    return masked_img


@app.get("/")
def home():
    content = """
<body>
<form action="/predict/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = file.file.read()
    decoded_image = cv2.imdecode(np.frombuffer(contents, np.uint8), -1)
    if not file:
        return {"msg":"No file sent"}

    res, im_png = cv2.imencode(".jpg", decoded_image)
    inference = io.BytesIO(im_png.tobytes())
    return StreamingResponse(content=inference, media_type="image/jpg")




if __name__ =="__main__":
    uvicorn.run(app)