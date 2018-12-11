# app.py - a minimal flask api using flask_restful
from flask import Flask
from flask_restful import Resource, Api
import os.path
import re
import sys
import tarfile
from flask import Flask, jsonify
from flask import make_response
from flask import request, render_template
from flask_bootstrap import Bootstrap
import numpy as np
from six.moves import urllib
import tensorflow as tf

from werkzeug import secure_filename

app = Flask(__name__)
api = Api(app)

@app.route("/classify", methods=["POST"])
def classify():
  results = []
  create_graph()
  url = request.form['url']
  imagery = urllib.request.urlopen(url)
  data = imagery.read()
  #response = request.get_data()
  print("Model loaded")
  (prediction_label, prediction_prob)  = run_inference_on_image(data)
  #print(predictions)
  results.append('(%s, %s)' % (str(prediction_prob), prediction_label))
  final_string = ' '.join(results)
  return final_string 

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile('intext.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image_data):
  with tf.Session() as sess:
    label_lines = [line.rstrip() for line
                            in tf.gfile.GFile("labels.txt")]
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)
    num_top_predictions = 5
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    results = []
    for node_id in top_k:
      human_string = label_lines[node_id]
      score = predictions[node_id]
      results.append((human_string, score))

    return results


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
