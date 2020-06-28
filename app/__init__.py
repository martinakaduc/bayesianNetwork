from flask import Flask

UPLOAD_FOLDER = './app/data'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.debug = True
from app import main
