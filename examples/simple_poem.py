from bespokelabs import curator

# Use GPT-4o-mini for this example.
llm = curator.SimpleLLM(model_name="gpt-4o-mini")

poem = llm("Write a poem about the bitter lesson in AI and keep it 100 words or less.")
print(poem)

# Use Claude 3.5 Sonnet for this example.
llm = curator.SimpleLLM(model_name="claude-3-5-sonnet-20240620", backend="litellm")

poem = llm("Write a sonnet about the bitter lesson in AI and make it visual.")
print(poem)
