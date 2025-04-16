from vllm import LLM, SamplingParams

prompts = [
    "Write an article about AI"
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="google/gemma-3-12b-it",
    tensor_parallel_size=4,
    dtype="float32",
    gpu_memory_utilization=0.9,  # Use 90% of available GPU memory
    max_model_len=16384  # Set a reasonable maximum sequence length
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")