from bottle import route, run, request, get
import bottle
import os
from detect_mrz import MRZRecognizer
from parse_mrz import MRZ

bottle.BaseRequest.MEMFILE_MAX = 10000 * 10000

@route('/hello')
def index():
    return "Hello World, how are you?"

@route('/upload', method='POST')
def do_upload():
    upload = request.files['image']
    name, ext = os.path.splitext(upload.filename)
    if ext not in ('.png','.jpg','.jpeg'):
        return 'File extension not allowed.'
    raw = upload.file.read()     
    text = MRZRecognizer.apply(raw)
    parsed_text = MRZ(text)
    json = parsed_text.to_dict()
    return json

run(host='localhost', port=8085)