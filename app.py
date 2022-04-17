from flask import Flask
from flask import request
import urllib.parse

app = Flask(__name__)

@app.route("/")
def hello():
  image = request.args.get('image')
  user = request.args.get('user')
  # print("====================")
  # print(imageURL)
  textInput = request.args.get('text')

  # imageURL = "\"" + "https://" + image + "\""
  x = image.split("/"+user+"/")

  temp = "https://" + x[0] + "%2F" + user+ "%2F" +x[1]
  imageURL = urllib.parse.quote(temp, safe="%/:=&?~#+!$,;'@()*[]")
  # print(imageURL)
  # print("====================")
  emotion = faceAnalyze(imageURL)
  # print(emotion)
  text = sentimentAnalyze(textInput) # change in future
  # print(text)
  if (((emotion == "sad") | (emotion == "angry")) & (text == 1)):
    # print(1)
    return "1"
  # print(0)
  return "0"


# import numpy as np
# import onnx
# import os
# import glob
# import onnx as backend

# from onnx import numpy_helper

# def load_model():
#   model = onnx.load('model.onnx')
#   test_data_dir = 'test_data_set_0'

#   # Load inputs
#   inputs = []
#   inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
#   for i in range(inputs_num):
#       input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
#       tensor = onnx.TensorProto()
#       with open(input_file, 'rb') as f:
#           tensor.ParseFromString(f.read())
#       inputs.append(numpy_helper.to_array(tensor))

#   # Load reference outputs
#   ref_outputs = []
#   ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
#   for i in range(ref_outputs_num):
#       output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
#       tensor = onnx.TensorProto()
#       with open(output_file, 'rb') as f:
#           tensor.ParseFromString(f.read())
#       ref_outputs.append(numpy_helper.to_array(tensor))

#   # Run the model on the backend
#   outputs = list(backend.run_model(model, inputs))

#   # Compare the results with reference outputs.
#   for ref_o, o in zip(ref_outputs, outputs):
#       np.testing.assert_almost_equal(ref_o, o)

from deepface import DeepFace
def faceAnalyze(imageURL):
  print(imageURL)
  obj = DeepFace.analyze(img_path = imageURL, actions = ['emotion'])
  return obj.get("dominant_emotion")


from textblob import TextBlob
def sentimentAnalyze(text):
  result = TextBlob(text).sentiment.polarity
  if result > 0.3:
    return 1
  if result < -0.3:
    return -1
  return 0

if __name__ == "__main__":
  app.run()