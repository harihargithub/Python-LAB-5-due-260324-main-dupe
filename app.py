from flask import Flask, request, render_template, flash
import numpy as np
import pickle
import logging

app = Flask(__name__)
app.secret_key = "super secret"

# Load the model
model = pickle.load(open("random_forest_model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get the data from the POST request.
    data = request.form

    # Make prediction using the model loaded from disk as per the data.
    prediction = model.predict([np.array(list(map(float, data.values())))])

    # Take the first value of prediction
    output = prediction[0]

    # Log the output
    logging.info("Prediction: %s", output)

    # Render the index.html template with the prediction result
    flash("Predicted resale Price is {}".format(output))
    return render_template("index.html")


if __name__ == "__main__":
    logging.basicConfig(filename="app.log", level=logging.INFO)
    print("Server running on http://127.0.0.1:5000")
    app.run(port=5000, debug=True)
