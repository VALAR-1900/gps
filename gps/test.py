from flask import Flask,render_template, jsonify, send_file, url_for

app = Flask(__name__)

@app.route('/download')
def download():    
    return send_file(url_for('static/images/caution.jpg') ,attachment_filename='result.jpg', filename_or_fp='caution.jpg',mimetype='image/jpg')

@app.route('/')
def home():
    return 'HELLO WORLD'

if __name__ == "__main__":
    app.debug = True
    app.run(port=4000)
