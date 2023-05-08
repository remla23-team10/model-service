# Model service

This is a Flask application that exposes an API to classify a given review as negative or positive. The app uses a machine learning model to classify the review. The model is fetched from https://github.com/remla23-team10/model-training.

## Requirements

You can install the required dependencies running:

```
pip install -r requirements.txt
```

## Usage

To run the application, use the following command:

```
gunicorn app:app
```

By default, the application will run on `http://localhost:8000`.

You can access the Swagger UI at `http://localhost:8000/apidocs`.

The API endpoint is `/predict`. To make a prediction, send a POST request to the `/predict` endpoint with a JSON body containing the review. The review should be sent under the `review` key in the JSON object. Here is an example:

```
POST http://localhost:8000/predict

{
  "review": { "This is a test review."
}
```

The API will return a JSON object with the predicted sentiment of the review:

```
{
    "result": "positive"
}
```

