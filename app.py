from flask import Flask, request, jsonify
import joblib
#from utils import clean_text

model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Hello, World!"})

@app.route('/predict', methods=['POST'])
def predict_from_json():
    try:
        data = request.get_json()
        if "text" not in data:
            return jsonify({"error": "Missing 'text' field in JSON request"}), 400

        text = data["text"]
        transformed_text = vectorizer.transform([text])

        dic = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}

        # Make prediction
        prediction = dic[model.predict(transformed_text)[0]]

        # Return response
        return jsonify({"prediction": str(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)