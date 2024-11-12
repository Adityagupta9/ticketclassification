from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import os

app = Flask(__name__)

# Load the pipeline model
def load_model(file_path):
    try:
        return joblib.load(file_path)
    except Exception:
        try:
            with open(file_path, "rb") as file:
                return pickle.load(file, encoding="latin1")
        except Exception as e:
            print(f"Model loading failed: {e}")
            return None

# Load the model (adjust the path as necessary)
model_path = os.path.join("model_pipeline", "pipeline.joblib")
model = load_model(model_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model loading failed."}), 500

    user_input = request.form.get("inquiry")
    if user_input:
        try:
            prediction = model.predict([user_input])
            return jsonify({"prediction": prediction[0]})
        except Exception as e:
            return jsonify({"error": f"Prediction error: {e}"}), 500
    else:
        return jsonify({"error": "Invalid input."}), 400

if __name__ == "__main__":
    app.run(debug=True)
