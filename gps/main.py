from flask import Flask,render_template, jsonify, send_file
import os
from random import sample
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

app = Flask(__name__)

@app.route('/') 
def home():
    return render_template('index.html')

@app.route('/enter')  
def submit():    
    import COPENF
    #import RAW_DATAF
    return '', 205

@app.route('/download')
def download():    
    #return send_file('static/download.xlsx',attachment_filename='Result',filename_or_fp='download')
    return send_file(os.path.join('static/', 'download.xlsx'), as_attachment=True, attachment_filename='Result.xlsx')
if __name__ == "__main__":
    app.debug = True
    app.run(port=6400)