from typing import List

import pandas as pd
from pydantic import BaseModel, Field

import bella
from bella import Prompter


class Subject(BaseModel):
    subject: str = Field(description="A subject")


class Subjects(BaseModel):
    subjects: List[Subject] = Field(description="A list of subjects")


class QA(BaseModel):
    question: str = Field(description="A question")
    answer: str = Field(description="An answer")


class QAs(BaseModel):
    qas: List[QA] = Field(description="A list of QAs")


subject_prompter = Prompter(
    prompt_func=lambda: {
        "user_prompt": f"Generate a diverse list of 3 subjects. Keep it high-level (e.g. Math, Science)."
    },
    model_name="gpt-4o-mini",
    response_format=Subjects,
)
result = subject_prompter()
subject_dataset = []
for subject in result:
    subject_dataset.extend(subject.subjects)


subsubject_prompter = Prompter(
    prompt_func=lambda subject: {
        "user_prompt": f"For the given subject {subject}. Generate 2 diverse subsubjects. No explanation."
    },
    model_name="gpt-4o-mini",
    response_format=Subjects,
)
result = subsubject_prompter(subject_dataset)
subsubject_dataset = []
for subject, subsubjects in zip(subject_dataset, result):
    subsubject_dataset.extend(
        [
            {"subject": subject.subject, "subsubject": subsubject.subject}
            for subsubject in subsubjects.subjects
        ]
    )

qa_prompter = Prompter(
    prompt_func=lambda subsubject: {
        "user_prompt": f"For the given subsubject {subsubject}. Generate 3 diverse questions and answers. No explanation."
    },
    model_name="gpt-4o-mini",
    response_format=QAs,
)
result = qa_prompter(subsubject_dataset)

qa_dataset = []
for subsubject, qas in zip(subsubject_dataset, result):
    qa_dataset.extend(
        [
            {
                "subject": subsubject["subject"],
                "subsubject": subsubject["subsubject"],
                "question": qa.question,
                "answer": qa.answer,
            }
            for qa in qas.qas
        ]
    )
print(pd.DataFrame.from_records(qa_dataset))
