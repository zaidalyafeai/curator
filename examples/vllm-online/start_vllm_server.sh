HOST=localhost
PORT=8787
MODEL_PATH="path/to/your/model"
CMD="
    vllm serve \
        ${MODEL_PATH} \
        --host ${HOST} \
        --port ${PORT} \
        --api-key token-abc123 \
        "
echo "Starting VLLM server on ${HOST}:${PORT}..."
bash -c "${CMD}"

sleep 20 # Wait for the server to start