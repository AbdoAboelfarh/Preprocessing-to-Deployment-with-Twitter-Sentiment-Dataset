from flask import Flask, request, jsonify
import joblib

# Load the saved pipeline
model = joblib.load("sentiment_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Twitter Sentiment Analysis API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Expecting JSON input {"text": "your tweet here"}
    
    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data["text"]
    prediction = model.predict([text])[0]
    
    return jsonify({"text": text, "sentiment": prediction})

if __name__ == "__main__":
    app.run(debug=True)
