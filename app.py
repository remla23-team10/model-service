import pickle
import joblib
from flasgger import Swagger
from flask import Flask, request

app = Flask(__name__)
swagger = Swagger(app)
cv = pickle.load(open("c1_BoW_Sentiment_Model.pkl", "rb"))
classifier = joblib.load('c2_Classifier_Sentiment_Model')

prediction_map = {
    0: "negative",
    1: "positive"
}


# TODO: implement preprocess
def preprocess(data):
    return cv.transform([data]).toarray()[0]


@app.route('/', methods=['POST'])
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
            required: sms
            properties:
                msg:
                    type: string
                    example: This is an example msg.
    responses:
      200:
        description: Some result
    """
    data = request.get_json()
    processed_input = preprocess(data)
    prediction = classifier.predict([processed_input])[0]
    return {
        "result": prediction_map[prediction],
    }


app.run(host="0.0.0.0", port=8080, debug=True)
