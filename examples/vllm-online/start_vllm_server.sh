source ../../../.vllm/bin/activate
export 
HOST=localhost
PORT=8787
MODEL_PATH="Qwen/Qwen2.5-32B-Instruct"
CMD="
    vllm serve \
        ${MODEL_PATH} \
        --host ${HOST} \
        --port ${PORT} \
        --api-key token-abc123 \
        --dtype half \
        --gpu-memory-utilization 0.95 \
        --tensor-parallel-size 4 \
        --max-model-len 16384 \
        --max_num_seqs 100
    "
echo "Starting VLLM server on ${HOST}:${PORT}..."
bash -c "${CMD}" &

sleep 60 # Wait for the server to start
