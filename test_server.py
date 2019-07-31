from flask import Flask, request, Response
import json 
import os
from foreground import fg
rect = (17,25,199,202)

app = Flask(__name__)
run = fg()


@app.route('/')
def display():
    return "1"

# /proposals?file=/home/user/Downloads/image.jpg
@app.route('/proposals')
def predict():
    if 'file' in request.args:
        image_path = request.args.get('file')
        # x1, y1, x2, y2 = request.args.get('x1'), request.args.get('y1'), request.args.get('x2'), request.args.get('y2')
        # coord = run.mod(image_path,rect = (int(x1), int(y1), int(x2), int(y2)))
        coord = run.mod(image_path,rect)

        flat = json.dumps(coord)

    return Response(flat, mimetype='application/json')

@app.route('/runner')
def call_runner():
    if 'arg_name' in request.args:
        k = os.system(request.args.get('arg_name'))
        if k == 0:
            resp ='{"status":200,"status_msg":"comm_execute success","data":1}'
    else:
        resp  = '{"status":200,"status_msg":"comm_execute fail","data":0}'
    return Response(resp, mimetype='text/json')

@app.errorhandler(500)
def err_manage(tmp):
    resp  = '{"status":200,"status_msg":"comm_execute fail","data":0}'
    return Response(resp, mimetype='text/json')
    
if __name__ =='__main__':
    app.run(debug = True, port = 5009)
