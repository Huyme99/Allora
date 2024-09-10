import json
from flask import Flask, Response
from model import download_data, format_data, train_model, get_inference
from config import model_file_path, TOKEN, TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER

app = Flask(__name__)

def update_data():
    """
    Orchestrates the data update process: download, format, train.
    """
    try:
        downloaded_files = download_data(TOKEN, TRAINING_DAYS, REGION, DATA_PROVIDER)
        format_data(downloaded_files, DATA_PROVIDER)
        train_model(TIMEFRAME)
        print("Data update and model training completed successfully.")  # Informative logging
    except Exception as e:
        print(f"Error during data update: {e}")  # Log errors for troubleshooting
        raise  # Re-raise the exception to propagate it to the calling function

@app.route("/inference/<string:token>")
def generate_inference(token):
    """
    Endpoint to generate an inference for a given token.
    """
    if not token or token.upper() != TOKEN:
        error_msg = "Token is required" if not token else "Token not supported"
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')

    try:
        inference = get_inference(token.upper(), TIMEFRAME, REGION, DATA_PROVIDER)
        return Response(json.dumps({"inference": inference}), status=200, mimetype='application/json') 
        # Return JSON for consistency and easier client-side handling
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

@app.route("/update")
def update():
    """
    Endpoint to trigger a data update and model retraining.
    """
    try:
        update_data()
        return Response(json.dumps({"status": "success"}), status=200, mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({"status": "error", "message": str(e)}), status=500, mimetype='application/json')

if __name__ == "__main__":
    try:
        update_data()  # Perform initial data update and training on startup
    except Exception as e:
        print(f"Error during initial data update: {e}")  # Log the error but continue

    app.run(host="0.0.0.0", port=8000)
