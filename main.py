# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 11:32:26 2022

@author: 91730
"""

# 1. Library imports
import numpy as np
import pydicom
# import pickle
from io import BytesIO
import uvicorn ##ASGI
from fastapi import FastAPI,File, UploadFile
from dataset import save_and_resize
from tensorflow.keras.models import load_model
from fastapi.responses import HTMLResponse
#from PIL import Image
from fastapi.staticfiles import StaticFiles
import os


# 2. Create the app object
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.post("/", response_class=HTMLResponse)
@app.get('/', response_class=HTMLResponse)
async def main():
    content = head_html + """
    <marquee width="800" behavior="alternate"><h1 style="color:red;font-family:Arial">Welcome To brain hemorrhage detection!</h1></marquee>
    <h3 style="font-family:Arial">We'll predict between hemorrhage and non hemorrhage on CT images and further classify it's type:</h3>
    """

    original_paths = ['ID_000a18499.png', 'ID_00e1db3ad.png']

    full_original_paths = ['static/' + x for x in original_paths]

    display_names = ['SDH', 'No Hemorrhage']

    column_labels = []

    content = content + '<img src = "/static/hemmtype.png" height ="180"/>'
    
    content = content + get_html_table1(full_original_paths, display_names, column_labels)

    content = content + """
    <h4 style="font-family:Arial">Please Upload Your Brain Hemorrhage Image, one at a time in dicom format</h4>
    <form  action="/uploadfiles/" enctype="multipart/form-data" method="post">
    <input name="file" type="file" multiple>
    <input type="submit">
    </form>
    </body>
    """
    return content

head_html = """
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
</head>
<body style="background-color:powderblue;">
<center>
"""

def get_html_table1(image_paths, names, column_labels):
    s = '<table align="center">'
    if column_labels:
        s += '<tr><th><h4 style="font-family:Arial">' + column_labels[0] + '</h4></th><th><h4 style="font-family:Arial">' + column_labels[1] + '</h4></th></tr>'
    
    for name, image_path in zip(names, image_paths):
        s += '<tr><td><img height="100" src="/' + image_path + '" ></td>'
        s += '<td style="text-align:center">' + name + '</td></tr>'
    s += '</table>'
    
    return s

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/Welcome')
def get_name(name1: str):
    return {'Welcome To brain hemorrhage detection API': f'{name1}'}

def read_file_as_image(data) -> np.ndarray:
    ds = BytesIO(data)
    return ds

@app.post("/uploadfiles/", response_class=HTMLResponse )
async def predict(file: UploadFile = File(...)):
    name = file.filename
    
    image = read_file_as_image(await file.read())
    image1 = pydicom.dcmread(image, force=True)
    img =save_and_resize(image1,name )
    image2 = np.resize(img,(1, 299, 299, 3))


    directory = os.getcwd()
    model_path = directory +'/static/effnetb5_101k.h5'
    print(model_path)
    model = load_model(model_path)
    val_preds = model.predict(image2)
    print('Original predicted value is', val_preds)

    y_pred = np.zeros([len(val_preds),6], dtype = int)
    for i in range(len(val_preds)):
        for j in range(6):
            if (val_preds[i,j] > 0.2):
                y_pred[i,j] = 1
        
    print(y_pred)

    types = ['epidural','intraparenchymal','intraventricular','subarachnoid','subdural']

    remar = []

    if y_pred[0,0] == 1:
        y = np.delete( y_pred[0], 0)
        index =0
        
        for i in y:
            if i==1:
                remark = types[index]
                remar.append(remark)
            index= index+1
        remar = ','.join(remar)
        msg = 'Uploaded file has brain hemorrhage and its type is ' + remar
            
    else: msg = 'Uploaded file has no brain hemorrhage'


    #pillow_image = Image.fromarray(img)
    #pillow_image.save('static/' + name)
    name = name.split('.')[0]
    name = name+ '.png'
    print('filename is', name)

    image_path = 'static/' + name

    table_html = get_html_table(msg,image_path, remar )

    content = head_html + """
    <marquee width="525" behavior="alternate"><h1 style="color:red;font-family:Arial">Here's Our Prediction!</h1></marquee>
    """ + str(table_html) + '''<br><form method="post" action="/">
    <button type="submit">Home</button>
    </form>'''

    return content

def get_html_table( msg, image_path,type):
    s = '<table align="center">'
    if msg:
        s += '<h4 style="font-family:Arial">' + str(msg) + '</h4>'
        s += '<tr><td><img height="120" src="/' + image_path + '" ></td>'
        s += '<td style="text-align:left">' + str(type) + '</td></tr>'
    s += '</table>'
    
    return s

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
#uvicorn main:app --reload
