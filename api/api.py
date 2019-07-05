import flask
from flask import request
import sys
# sys.path.insert(0, '../')
from model import QuestionClassifier
import numpy as np

app = flask.Flask(__name__)
# app.config["DEBUG"] = True

model = QuestionClassifier(ckp_dir=".")


def simpleClassifier(strLength, num_purchases):
    if strLength > 50 and num_purchases >= 10:
        return 1
    return 0


def fancyClassifier(strLength, num_purchases):
    data = np.expand_dims(np.asarray([strLength, num_purchases]), axis=0)
    dec, prob = model.predict(data)
    return dec, (prob-0.5)*2


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Question Classifier</h1>
<p>A prototype API for classifying user questions.</p>
<p>Example Calls:</p>
<p>Simple Model: http://ip/api/v1/QC/?ql=30&np=100&model=0</p>
<p>FC Model: http://ip/api/v1/QC/?ql=60&np=3000&model=1</p>
<p> where ip is either 127.0.0.1:5000 for running it locally or the ip of the docker virtual environment</p>'''


@app.route('/api/v1/QC/')
def api_id():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    reqKeys = ['ql', 'np']

    for key in reqKeys:
        if not key in request.args.to_dict().keys():
            return "Error: No " + key + " field provided.\
            Please specify " + key + "."

    var1 = request.args.get('ql', 1, type=int)
    var2 = request.args.get('np', 1, type=int)
    var3 = request.args.get('model', 0, type=int)

    if var1 < 1 or var2 < 0:
        return "Error: Question length must be at larger than one and\
                number of purchases must be non negative."
    if var3 == 0:
        return 'Decision: %s' % str(simpleClassifier(var1, var2))
    else:
        dec, prob = fancyClassifier(var1, var2)
        return 'Decision: %s Probability: %s' % (str(int(dec[0][0])), str(prob[0][0]))
    # return '<h1>Result: %s</h1>' % str(simpleClassifier(var1, var2))

app.run(port=5000, host='0.0.0.0')
