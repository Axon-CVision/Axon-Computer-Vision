from flask import Flask, request, Response, jsonify
# import jsonify
import numpy as np
import cv2

# Initialize the Flask application
app = Flask(__name__)


# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    image = request.files['image']
    filename = image.filename
    image.save(filename)
    print(type(image))

    # do some fancy processing here....

    # build a response dict to send back to client
    return jsonify({'message': 'Image uploaded successfully'})


# start flask app
app.run(host="0.0.0.0", port=5000)
