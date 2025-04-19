cd /ibex/ai/home/alyafez/curator/custom/examples
export 
HOST=localhost
PORT=8787
MODEL_NAME=gemma-3-27b-it
MODEL_PATH="/ibex/ai/home/alyafez/curator/examples/custom/${MODEL_NAME}"
CURATOR_VIEWER=1
VLLM_CONFIGURE_LOGGING=0
LANGUAGE=eng-Latin
CMD="
    vllm serve \
        ${MODEL_PATH} \
        --host ${HOST} \
        --port ${PORT} \
        --gpu-memory-utilization 0.95 \
        --tensor-parallel-size 8 \
        --pipeline-parallel-size 1 \
        --max-model-len 16384 \
        --dtype float32 \
        --max-num-seqs 100 \
        --disable-log-requests \
	    --served-model-name ${MODEL_NAME}
    "
echo "Starting VLLM server on ${HOST}:${PORT}..."
bash -c "${CMD}" & python score.py --mode vllm-online --model ${MODEL_NAME} --language ${LANGUAGE}
bash killall.sh
exit 0
