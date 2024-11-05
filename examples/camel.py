from typing import List

import pandas as pd
from pydantic import BaseModel, Field

import bella
import prompt


class Subject(BaseModel):
    subject: str = Field(description="A subject")


class Subjects(BaseModel):
    subjects: List[Subject] = Field(description="A list of subjects")


class QA(BaseModel):
    question: str = Field(description="A question")
    answer: str = Field(description="An answer")


class QAs(BaseModel):
    qas: List[QA] = Field(description="A list of QAs")


result = bella.completions(
    (),
    prompter=prompt.Prompter(
        system_prompt="You are a helpful AI assistant.",
        user_prompt="Generate a diverse list of 3 subjects. Keep it high-level (e.g. Math, Science).",
        response_format=Subjects,
        model_name="gpt-4o-mini",
    ),
)
subject_dataset = []
for subject in result:
    subject_dataset.extend(subject.subjects)
print(pd.DataFrame.from_records(subject_dataset))

result = bella.completions(
    dataset=subject_dataset,
    prompter=prompt.Prompter(
        system_prompt="You are a helpful AI assistant.",
        user_prompt="For the given subject {{ subject }}. Generate 3 diverse subsubjects. No explanation.",
        response_format=Subjects,
        model_name="gpt-4o-mini",
    ),
)
subsubject_dataset = []
for subject, subsubjects in zip(subject_dataset, result):
    for subsubject in subsubjects.subjects:
        subsubject_dataset.append(
            {"subject": subject.subject, "subsubject": subsubject.subject}
        )
print(pd.DataFrame.from_records(subsubject_dataset))

result = bella.completions(
    dataset=subsubject_dataset,
    prompter=prompt.Prompter(
        system_prompt="You are a helpful AI assistant.",
        user_prompt="For the given subject {{ subsubject }}, generate 10 diverse questions and answers. No explanation.",
        response_format=QAs,
        model_name="gpt-4o-mini",
    ),
)
qa_dataset = []
for subsubject, qas in zip(subsubject_dataset, result):
    for qa in qas.qas:
        qa_dataset.append(
            {
                "subject": subsubject["subject"],
                "subsubject": subsubject["subsubject"],
                "question": qa.question,
                "answer": qa.answer,
            }
        )
print(pd.DataFrame.from_records(qa_dataset))
