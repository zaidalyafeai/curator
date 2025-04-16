import requests
import time
from requests.exceptions import ConnectionError, RequestException
from urllib3.exceptions import NewConnectionError, MaxRetryError
from openai import OpenAI

HOST = "localhost"
PORT = 8000
MAX_RETRIES = 30  # 5 minutes with 10 second intervals
RETRY_INTERVAL = 10

def check_server_status(host, port):
    url = f"http://{host}:{port}/v1"
    try:
        client = OpenAI(
            api_key="EMPTY",
            base_url=url,
        )
        print("running inference")
        chat_response = client.chat.completions.create(
            model="gemma-3-4b-it",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is ai?"},
            ]
        )
        print("inference done")
        print(chat_response)
        return True
    except ConnectionError as e:
        # This catches both ConnectionRefusedError and other connection issues
        if isinstance(e.args[0], MaxRetryError) and isinstance(e.args[0].reason, NewConnectionError):
            # This is the specific case of connection refused
            return False
        print(f"Connection error: {e}")
        return False
    except RequestException as e:
        print(f"Request error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

retry_count = 0
while not check_server_status(HOST, PORT):
    if retry_count >= MAX_RETRIES:
        print("Maximum retries reached. Server failed to start.")
        exit(1)
    print(f"Waiting for the server to start... (Attempt {retry_count + 1}/{MAX_RETRIES})")
    time.sleep(RETRY_INTERVAL)
    retry_count += 1

print("Server is running")
