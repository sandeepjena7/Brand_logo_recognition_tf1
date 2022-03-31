
from fastapi import FastAPI ,Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from fastapi.responses import HTMLResponse,FileResponse,JSONResponse
from typing import Optional,List,Union,Dict
from pathlib import Path
from utils.all_utills import ClientImageInput,ClientImageOutput,error_handel_user_images
from research.prediction import BrandsLog


app  = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST","GET"], 
    allow_headers=["*"],
    max_age=2 # how mcuh hit api per second
    )

class NotEncodeBase64(Exception):
    def __init__(self,message:str=None):
        self.message = message

class ImageIsNotOpening(Exception):
    def __init__(self,message:str=None):
        self.message = message

@app.exception_handler(NotEncodeBase64)
def not_encode_base64(request:Request,exc:NotEncodeBase64):
    
    return JSONResponse(
        status_code=418
        ,content = {"message":f"{exc.message} "}
    )

@app.exception_handler(ImageIsNotOpening)
def image_not_open(request:Request , exc:ImageIsNotOpening):
    
    return JSONResponse(
        status_code=418
        ,content = {"message":f"{exc.message}"}
    )




class ClientApp(BrandsLog):
    def __init__(self,Path_Ckpt:Path,labelmap_ph:Path):
        super(ClientApp, self).__init__(Path_Ckpt,labelmap_ph)


clApp = ClientApp("prediction_service\\save_model\\frozen_inference_graph.pb","prediction_service\\labelmap\\labelmap.pbtxt")


# templates = Jinja2Templates(directory="webapp/templates")

# @app.get("/",response_class=HTMLResponse)
# def read(request:Request):
#     return templates.TemplateResponse("index.html",{"request":request})


@app.post("/predict",response_model=List[Union[ClientImageOutput,ClientImageInput]])
def predict(file:ClientImageInput):
    s = file.image
    if not isinstance(file.image ,bytes):
        raise NotEncodeBase64(message="image not in enocde bytes format" )
    elif isinstance(file.image,bytes):
        try:
            error_handel_user_images(file.image)
        except :
            raise ImageIsNotOpening(message="image is Not opening")

    clApp.base64toimage = file.image
    output = clApp.getPredictions()
    return output
    
    # return {'sandee':6}
    
if __name__ == "__main__":
    uvicorn.run(app,port=8080)