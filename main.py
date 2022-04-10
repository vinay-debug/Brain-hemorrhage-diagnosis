# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 11:32:26 2022

@author: 91730
"""

# 1. Library imports
import numpy as np
import pydicom
import pickle
from io import BytesIO
import uvicorn ##ASGI
from fastapi import FastAPI,File, UploadFile
from dataset import save_and_resize
from keras.models import load_model


# 2. Create the app object
app = FastAPI()


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Lets get started'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/Welcome')
def get_name(name: str):
    return {'Welcome To brain hemorrhage detection API': f'{name}'}

def read_file_as_image(data) -> np.ndarray:
    ds = BytesIO(data)
    return ds

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image1 = pydicom.dcmread(image, force=True)
    img =save_and_resize(image1)
    model=pickle.load(open('model3.pkl','rb'))
    
    #model.summary()
    #model = load_model('D:\project\effnetb5_101k.h5')
    val_preds = model.predict(img)
    print('Original predicted value is', val_preds)

    y_pred = np.zeros([len(val_preds),6], dtype = int)
    for i in range(len(val_preds)):
        for j in range(6):
            if (val_preds[i,j] > 0.2):
                y_pred[i,j] = 1
        
    print(y_pred)

    type = ['epidural','intraparenchymal','intraventricular','subarachnoid','subdural']

    if y_pred[0,0] == 1:
        y = np.delete( y_pred[0], 0)
        index =0
        remar = []
        for i in y:
            if i==1:
                remark = type[index]
                remar.append(remark)
            index= index+1
        msg =('Uploaded file has brain hemorrhage and its type is', remar)
            
    else: msg =('Uploaded file has no brain hemorrhage')

    return {
        'message': msg,
    }
# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
#uvicorn main:app --reload
