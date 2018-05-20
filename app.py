import flask
import keras
import numpy as np
import PIL
import tensorflow as tf

app = flask.Flask(__name__)
model = keras.models.load_model('digit_model.h5')
graph = tf.get_default_graph()

@app.route('/')
def index():
    inst = '''
    curl -F 'file=@/path/to/digit_pic.png' {}
    '''.format(flask.url_for('predict', _external=True))
    return inst


@app.route('/status')
def status():
    return flask.jsonify({'status': 'Ok'})


@app.route('/predict', methods=['POST'])
def predict():
    fp = flask.request.files['file']
    image = PIL.Image.open(fp)
    image = np.asarray(image.resize((56, 56)))
    image = image.reshape(1, 56, 56, 1)

    with graph.as_default():
        prediction = model.predict(image)

    digit = np.argmax(prediction)
    return flask.jsonify({'digit': int(digit)})


if __name__ == '__main__':
    app.run(host='0.0.0.0')
