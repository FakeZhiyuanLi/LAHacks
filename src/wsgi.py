from flask import Flask, render_template, request, session, redirect, url_for
from flask_session import Session
import ImagePrediction
import torch
import os

app = Flask(__name__)

app.secret_key = '4728642b0797e5cc778bbe1651c539dcc24b605c639270fcf8ba67e4214f3c84ee954acc24608ae389408fb475cd8e448f7907f1ed1b48711697a6f955ac69acc785dcc4240e0fcb979001b918d0b6962d947c60cef77d5fa3b2cdc10fdeb54521e41211'
app.config['SESSION_TYPE'] = 'filesystem'

# model = torch.jit.load(os.path.join("models", "model.pt"))
# model.eval()

Session(app)

def getObject(image_path):
    prediction = ImagePrediction.predict_image(image_path)
    return prediction, ImagePrediction.is_recyclable(prediction)

@app.route('/')
def index():
    return render_template('home.html')

if __name__ == '__main__':
    app.run()
    # print(ImagePrediction.predict_image("/Users/zhiyuan/Desktop/ThomasTheDankEngineCode/Python/LAHacks/uploaded_files/metal357.jpg"))

    