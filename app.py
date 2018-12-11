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
from flask_cors import CORS
import numpy as np
import urllib
import time
from PIL import Image
import tensorflow as tf
import boto3

from werkzeug import secure_filename

app = Flask(__name__)
cors = CORS(app)
#api = Api(app)
def write_to_db(guid,category,percentage,time,mvId,year,make,model,trim):
  dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
  table = dynamodb.Table('MachineLearningAnglePrediction')
  response = table.put_item(
   Item={
        'Guid': guid,
        'Category': category,
        'Percentage': str(percentage),
        'Time': str(time),
	'MasterVehicleId' : mvId,
	'Year' : year,
	'Make' : make,
	'Model' : model,
	'Trim' : trim,
   }
  )


@app.route("/classify", methods=["POST"])
#@cross_origin()
def classify():
  url  = request.form['url']
  #data = all_data.split(';')
  #MasterVehicleID = data[0]
  #guid = data[1]
  #year = data[2]
  #make = data[3]
  #model = data[4]
  #trim = data[5]
  #condition = data[6]

  #url = "https://content.homenetiol.com/distributed-images/" + guid
  #response = request.get_data()
  time1 = time.time()
  results_left = run_inference_on_image(url,"left")
  if results_left[0][1] < 0.9 :
    results_right = run_inference_on_image(url,"right")
    if results_left[0][1] > results_right[0][1]:
      time2 = time.time()
      #write_to_db(guid,results_left[0][0],results_left[0][1],time2-time1,MasterVehicleID, year, make,model,trim)  
      return str(results_left[0])
    else:
      time2 = time.time()
      #write_to_db(guid,results_right[0][0],results_right[0][1],time2-time1,MasterVehicleID, year, make,model,trim)
      return str(results_right[0])
  else:
    time2 = time.time()
    return str(results_left[0])
    #write_to_db(guid,results_left[0][0],results_left[0][1],time2-time1,MasterVehicleID, year, make,model,trim)
  #print(predictions)
  #results.append('(%s, %s)' % (str(prediction_label), prediction_prob))
  #final_string = ' '.join(results)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


def run_inference_on_image(url,side):
  input_layer = "Placeholder"
  output_layer = "final_result"
  img = Image.open(urllib.urlopen(url))
  width, height = img.size
  if side == "left":
    graph = app.graph_left
    w = width/2
    imCrop = img.crop((0, 0, w , height))
  else:
    graph = app.graph_right
    w = width/2
    imCrop = img.crop((w, 0, width , height))
  label_lines = [line.rstrip() for line
                            in tf.gfile.GFile("output.txt")]
  t = read_tensor_from_image_file(
      imCrop,299,299,0,255)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  with tf.Session(graph=graph) as sess:
    pred = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
    })
  predictions = np.squeeze(pred)

  top_k = predictions.argsort()[-8:][::-1]
  #labels = load_labels(label_file)
  results = []
  for node_id in top_k:
    human_string = label_lines[node_id]
    score = predictions[node_id]
    results.append((human_string, score))

  return results


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(image_data,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  #file_reader = tf.read_file(file_name, input_name)
  #image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_data, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
   label.append(l.rstrip())
   return label

app.graph_left=load_graph('./half.pb') 
app.graph_right=load_graph('./righthalf.pb')
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
