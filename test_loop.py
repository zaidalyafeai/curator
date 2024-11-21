from datasets import Dataset

from bespokelabs import curator

ds = Dataset.from_dict({"i": [0]})

print("SHOULD CACHE since we're using the same value in a loop")
for x in [1,1,1]:

    def prompt_func():
        print(f"x is {x}")
        return f"Say {x}. Do not explain."

    def add_x(row):
        row["i"] = row["i"] + x
        return row

    topic_generator = curator.Prompter(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
    )
    print(topic_generator().to_pandas())

print("SHOULD NOT CACHE since we're using different values in a loop")
for x in [1, 2, 3]:

    def prompt_func():
        print(f"x is {x}")
        return f"Say {x}. Do not explain."

    def add_x(row):
        row["i"] = row["i"] + x
        return row

    topic_generator = curator.Prompter(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
    )
    print(topic_generator().to_pandas())
