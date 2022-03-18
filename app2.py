from flask import Flask, jsonify, request
from classify2 import getPrediction

app = Flask(__name__)

@app.route("/predict-alphabet", methods=["POST"])
def predict_data():
    image = request.files.get("alphabet")
    prediction2 = getPrediction(image)
    return jsonify({"prediction": prediction2}), 200

if __name__ == "__main__":
    app.run(debug=True)