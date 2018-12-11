# app.py - a minimal flask api using flask_restful

import os.path
import re
import sys
import tarfile
import numpy as np
from six.moves import urllib
import tensorflow as tf
import time
import boto3
import cPickle as pickle

def write_to_db(url,result):
  dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
  table = dynamodb.Table('MachineLearningImages')
  response = table.put_item(
   Item={
        'MasterVehicleId': '1235',
        'IntExt': result,
        'URL': url,
   }
  )

class Stopwatch:

    # Construct self and start it running.
    def __init__(self):
        self._creationTime = time.time()  # Creation time

    # Return the elapsed time since creation of self, in seconds.
    def elapsedTime(self):
        return time.time() - self._creationTime

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  graph = tf.Graph()

  with tf.gfile.FastGFile('intext.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
  return graph

def run_inference_on_image(image_data, graph):
  config = tf.ConfigProto()
  config.intra_op_parallelism_threads = 10
  config.inter_op_parallelism_threads = 10
  saver = tf.train.Saver()
  with tf.Session(graph = graph) as sess:
    label_lines = [line.rstrip() for line
                            in tf.gfile.GFile("labels.txt")]
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    saver.save(session, 'car-model')
    predictions = np.squeeze(predictions)
    num_top_predictions = 5
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    results =""
    for node_id in top_k:
      human_string = label_lines[node_id]
      score = predictions[node_id]
      if score > 0.9:
        results = human_string
      
  sess.close()  
  return results



def classify():
  t1 = time.time()
  final_string = ""
  #if os.path.isfile('mypickle.pickle'):
   # with open('mypickle.pickle') as f:
    #  graph = pickle.load(f)
    #final_string = everything_else(graph)
  #else:
  graph = create_graph()
  #with open('mypickle.pickle', 'wb') as f:
   #   pickle.dump(graph, f)
   # f.flush()
   # text = 'creating_graph';
  final_string = everything_else(graph)
  t2 = time.time()
  #response = request.get_data()

  return final_string


def everything_else(graph):
  results = []
  t3 = time.time()
  url = sys.argv[1]
  imagery = urllib.request.urlopen(url)
  data = imagery.read()
  t4 = time.time()
  #print("Model loaded")
  t5 = time.time()
  (prediction_label, prediction_prob)  = run_inference_on_image(data,graph)
  #print(predictions)
  t6 = time.time()
  #results.append('(%s, %s)' % (str(prediction_prob), prediction_label))
  results.append('(%s, %s, %s, %s)' % (str(t2-t1),str(t4-t3),str(t6-t5), text))
  final_string = ' '.join(results)


t1 = time.time()
final_string = ""
#if os.path.isfile('mypickle.pickle'):
#  with open('mypickle.pickle') as f:
 #   graph = pickle.load(f)
  #  final_string = everything_else(graph)
#else:
graph = create_graph()
  #with open('mypickle.pickle', 'wb') as f:
   # pickle.dump(graph, f)
  #text = 'creating_graph';
final_string = everything_else(graph)
t2 = time.time()
print(t2-t1)

#results = []
#now1=time.time()
#graph = create_graph()
#then1=time.time()
#url = sys.argv[1]
#now2 = time.time()
#imagery = urllib.request.urlopen(url)
#data = imagery.read()
#then2=time.time()
#response = request.get_data()
#now3=time.time()
#results  = run_inference_on_image(data, graph)
#print(results)
#then3=time.time()
#print(then1-now1)
#print(then2-now2)
#print(then3-now3)
#results.append('(%s, %s)' % (str(prediction_prob), prediction_label))
#final_string = ' '.join(results)
#print(results) 

