from bespokelabs.curator import LLM
import tempfile
import huggingface_hub
import pytest
import shutil
import time
import os
import socket


def download_model(model_name):
    """Download a model from the Hugging Face Hub."""
    tmpdirname = tempfile.mkdtemp()
    model_path = huggingface_hub.snapshot_download(
        repo_id=model_name, repo_type="model", local_dir=tmpdirname
    )
    return model_path


def start_vllm_server(model_path, host, port):
    """Start the VLLM server."""
    cmd = f"vllm serve {model_path} --host={host} --port={port} --api-key=token-abc123 &"
    try:
        os.system(cmd)
    except Exception as e:
        print(e)
        raise e


@pytest.mark.parametrize("model_name", ["HuggingFaceTB/SmolLM-135M-Instruct"])
def test_online_vllm(model_name):

    model_path = download_model(model_name)
    host = socket.gethostname().split(".")[0]

    port = 5432

    start_vllm_server(model_path, host, port)

    time.sleep(60)

    url = f"http://{host}:{port}/v1/chat/completions"

    prompter = LLM(
        prompt_func=lambda row: "write me a poem",
        model_name=model_path,
        url=url,
        backend="openai",
        api_key="token-abc123",
    )

    dataset = prompter()
    shutil.rmtree(model_path)
    assert dataset is not None
    assert len(dataset) == 1
    assert "response" in dataset.column_names
