import base64
import io
from PIL import Image
import cv2
import numpy as np
from pydantic import BaseModel
from pathlib import Path
import yaml

def error_handel_user_images(base64bytes:str):

    base64str = base64.b64decode(base64bytes)
    bytesObj = io.BytesIO(base64str)
    img = Image.open(bytesObj)
    img_cv = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)



class ClientImageInput(BaseModel):
    image: bytes
    # threshold: Optional[float] = 0.4


class ClientImageOutput(BaseModel):
    className:str  
    confidence:str  
    yMin:str
    xMin:str
    yMax:str
    xMax:str


    
def read_yaml(filename:Path) -> dict:

    with open(filename , 'r') as config_file:
        data = yaml.safe_load(config_file)
    
    return data
