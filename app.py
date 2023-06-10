from urllib.request import urlopen

import os
import joblib
from flasgger import Swagger
from flask import Flask, request

import pandas as pd
from restaurant_preprocessing import Preprocessing

app = Flask(__name__)
swagger = Swagger(app)

classifier_url = stock_url = os.environ['CLASSIFIER_URL']
vectorizer_url = stock_url = os.environ['VECTORIZER_URL']

classifier = joblib.load(urlopen(classifier_url))
preprocesser = Preprocessing()
preprocesser.vectorizer_from_url(vectorizer_url)

prediction_map = {
    0: "negative",
    1: "positive"
}


def preprocess(data):
    msg = data['review']
    preprocessed = preprocesser.preprocess_review(msg)
    return preprocesser.transform([preprocessed])

def getTrueSentiment(data):
    if (data['sentiment'] == 'positive' and data['correct']) or (data['sentiment'] == 'negative' and not data['correct']):
      return 1 
    else:
      return 0

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

@app.put('/incrementalTrain')
def incremental_train():
    """
    Retrain the model to add new data, using partial_fit(), which retains previously learned features.
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: array
            items:
              type: object
              properties:
                review:
                  type: string
                  example: This is an example review.
                sentiment:
                  type: string
                  enum: [negative, positive]
                  example: negative
                correct:
                  type: boolean
                  example: false
    responses:
      204:
        description: model updated successfully
    """
    data = request.get_json()

    reviews = list(map(lambda x: x['review'], data))
    sentiments = list(map(getTrueSentiment, data))

    dataset = pd.DataFrame(data={'Review': reviews, 'liked': sentiments})

    corpus = preprocesser.preprocess_dataset(dataset)
    X = preprocesser.transform(corpus)
    y = dataset.iloc[:, 1].values

    classifier.partial_fit(X, y)

    return ('', 204)