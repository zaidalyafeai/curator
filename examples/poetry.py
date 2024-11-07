from bespokelabs import curator

poet = curator.Prompter(
    prompt_func=lambda: {
        "user_prompt": "Write a poem about the beauty of computer science"
    },
    model_name="gpt-4o-mini",
)

poem = poet()
print(poem.to_list()[0])
