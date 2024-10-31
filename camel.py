import asyncio
import prompt

from pydantic import BaseModel, Field

from bella import Dataset, ListModel


class Subject(BaseModel):
    subject: str


class QA(BaseModel):
    question: str = Field(description="A question")
    answer: str = Field(description="A answer")


GetSubjects = prompt.Prompter(
    system_prompt="You are a helpful AI assistant.",
    user_prompt="Generate a diverse list of 1 subjects. Keep it high-level (e.g. Math, Science).",
    response_format=ListModel[Subject],
    model_name="gpt-4o-mini",
)


GetSubSubjects = prompt.Prompter(
    system_prompt="You are a helpful AI assistant.",
    user_prompt="For the given subject {{ subject.subject }}. Generate 3 diverse subsubjects. No explanation.",
    response_format=ListModel[Subject],
    model_name="gpt-4o-mini",
)


GetQAList = prompt.Prompter(
    system_prompt="You are a helpful AI assistant.",
    user_prompt="For the given subject {{ subsubject.subject }}, generate 1 diverse questions and answers. No explanation.",
    response_format=ListModel[QA],
    model_name="gpt-4o-mini",
)


def camelai():
    # Generate initial subjects.
    subject_dataset = Dataset.empty().completions(
        prompter=GetSubjects,
        output_column="subject",
        name="Generate subjects",
    ).flatten()

    print(subject_dataset)
    print(subject_dataset[0])

    # Generate subsubjects.
    subsubject_dataset = subject_dataset.completions(
        prompter=GetSubSubjects,
        output_column="subsubject",
        name="Generate sub-subjects",
    ).flatten()

    print(subsubject_dataset)
    print(subsubject_dataset[0])
    breakpoint()

    # Generate list of QA pairs.
    qa_dataset = subsubject_dataset.completions(
        prompter=GetQAList, output_column="qa", name="Generate QAs"
    ).flatten()

    print(qa_dataset)
    print(qa_dataset[0])

    return qa_dataset


camelai()
