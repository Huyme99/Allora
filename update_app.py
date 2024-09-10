import os
import requests

# Retrieve the inference API address from the environment variable
inference_address = os.environ.get("INFERENCE_API_ADDRESS")

if not inference_address:
    print("Error: INFERENCE_API_ADDRESS environment variable not set.")
    exit(1)  # Exit with an error code

url = f"{inference_address}/update"

print("UPDATING INFERENCE WORKER DATA")

try:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes (4xx and 5xx)

    # Request was successful
    content = response.json()  # Assuming the response is JSON

    if content.get("status") == "success":
        print("Data update and model training successful.")
        exit(0)  # Exit with success code
    else:
        error_message = content.get("message", "Unknown error during update")
        print(f"Error during update: {error_message}")
        exit(1)

except requests.exceptions.RequestException as e:
    # Handle any exceptions that occur during the request
    print(f"Request failed: {e}")
    exit(1)
