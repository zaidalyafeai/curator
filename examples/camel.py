from typing import List

from pydantic import BaseModel, Field

import bella
import prompt


class Subjects(BaseModel):
    subjects: List[str] = Field(description="A list of subjects")


class QA(BaseModel):
    question: str = Field(description="A question")
    answer: str = Field(description="A answer")


class QAs(BaseModel):
    qas: List[QA] = Field(description="A list of QAs")


GetSubjects = prompt.Prompter(
    system_prompt="You are a helpful AI assistant.",
    user_prompt="Generate a diverse list of 3 subjects. Keep it high-level (e.g. Math, Science).",
    response_format=Subjects,
    model_name="gpt-4o-mini",
)


GetSubSubjects = prompt.Prompter(
    system_prompt="You are a helpful AI assistant.",
    user_prompt="For the given subject {{ subjects.subjects }}. Generate 3 diverse subsubjects. No explanation.",
    response_format=Subjects,
    model_name="gpt-4o-mini",
)


GetQAList = prompt.Prompter(
    system_prompt="You are a helpful AI assistant.",
    user_prompt="For the given subject {{ subsubjects.subjects }}, generate 1 diverse questions and answers. No explanation.",
    response_format=QAs,
    model_name="gpt-4o-mini",
)


def camelai():
    subject_dataset = bella.flatten_list(
        bella.completions(
            dataset=bella.empty(),
            prompter=GetSubjects,
            output_column="subjects",
            name="Generate subjects",
        ).flatten()
    )

    print(subject_dataset[0])
    print(subject_dataset[1])
    print(subject_dataset[2])

    # Generate subsubjects.
    subsubject_dataset = bella.flatten_list(
        bella.completions(
            dataset=subject_dataset,
            prompter=GetSubSubjects,
            output_column="subsubjects",
            name="Generate sub-subjects",
        ).flatten()
    )

    print(subsubject_dataset[0])
    print(subsubject_dataset[1])
    print(subsubject_dataset[2])

    # Generate list of QA pairs.
    qa_dataset = bella.flatten_list(
        bella.completions(
            dataset=subsubject_dataset,
            prompter=GetQAList,
            output_column="qa",
            name="Generate QAs",
        ).flatten()
    ).flatten()

    print(qa_dataset[0])
    print(qa_dataset[1])
    print(qa_dataset[2])

    return qa_dataset


camelai()
