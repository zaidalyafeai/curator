import asyncio

from pydantic import BaseModel, Field

from bella import Dataset, ListModel


class Subject(BaseModel):
    subject: str


class QA(BaseModel):
    question: str = Field(description="A question")
    answer: str = Field(description="A answer")


async def camelai(model):
    # Generate initial subjects.
    subject_dataset = (
        await Dataset.empty().completions(
            model_name=model,
            system_prompt="You are a helpful AI assistant.",
            user_prompt="Generate a diverse list of 1 subjects. Keep it high-level (e.g. Math, Science).",
            response_format=ListModel[Subject],
            output_column="subject",
            name="Generate subjects",
        )
    ).flatten()

    # Generate subsubjects.
    subsubject_dataset = (
        await subject_dataset.completions(
            model_name=model,
            system_prompt="You are a helpful AI assistant.",
            user_prompt="For the given subject {{ subject.subject }}. Generate 3 diverse subsubjects. No explanation.",
            response_format=ListModel[Subject],
            output_column="subsubject",
            keep_columns=True,
            name="Generate sub-subjects",
        )
    ).flatten()

    # Generate QA pairs.
    qa_dataset = (
        await subsubject_dataset.completions(
            model_name=model,
            system_prompt="You are a helpful AI assistant.",
            user_prompt="For the given subject {{ subsubject.subject }}, generate 1 diverse questions and answers. No explanation.",
            response_format=ListModel[QA],
            output_column="qa",
            keep_columns=True,
            name="Generate QAs",
        )
    ).flatten()

    qa_dataset.display()
    return qa_dataset


asyncio.run(camelai("gpt-4o-mini"))
