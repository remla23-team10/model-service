from urllib.request import urlopen

import joblib
from flasgger import Swagger
from flask import Flask, request

from preprocessing import Preprocessing

app = Flask(__name__)
swagger = Swagger(app)
classifier = joblib.load(
    urlopen('https://github.com/remla23-team10/model-training/raw/main/Classifier_Sentiment_Model'))
preprocesser = Preprocessing()
preprocesser.vectorizer_from_url('https://github.com/remla23-team10/model-training/raw/main/BoW_Vectorizer.pkl')

prediction_map = {
    0: "negative",
    1: "positive"
}


def preprocess(data):
    msg = data['review']
    preprocessed = preprocesser.preprocess_review(msg)
    return preprocesser.transform([preprocessed])


@app.post('/predict')
def predict():
    """
    Make a prediction
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            properties:
              review:
                type: string
                example: This is an example msg.
    responses:
      200:
        description: prediction of review being positive or negative
        schema:
          type: object
          properties:
            sentiment:
              type: string
              enum: [negative, positive]
              example: negative
    """
    data = request.get_json()
    processed_input = preprocess(data)
    prediction = classifier.predict(processed_input)[0]
    return {
        "sentiment": prediction_map[prediction],
    }