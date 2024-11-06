from typing import List

import pandas as pd
from pydantic import BaseModel, Field

import bella
from prompt import prompter


class Subject(BaseModel):
    subject: str = Field(description="A subject")


class Subjects(BaseModel):
    subjects: List[Subject] = Field(description="A list of subjects")


class QA(BaseModel):
    question: str = Field(description="A question")
    answer: str = Field(description="An answer")


class QAs(BaseModel):
    qas: List[QA] = Field(description="A list of QAs")


@prompter("gpt-4o-mini", Subjects)
def get_subjects():
    return {
        "user_prompt": f"Generate a diverse list of 3 subjects. Keep it high-level (e.g. Math, Science)."
    }


@prompter("gpt-4o-mini", Subjects)
def get_subsubjects(subject):
    return {
        "user_prompt": f"For the given subject {subject}. Generate 3 diverse subsubjects. No explanation."
    }


@prompter("gpt-4o-mini", QAs)
def get_qas(subsubject):
    return {
        "user_prompt": f"For the given subject {subsubject}, generate 3 diverse questions and answers. No explanation."
    }


result = bella.completions(prompter=get_subjects)
subject_dataset = []
for subject in result:
    subject_dataset.extend(subject.subjects)

result = bella.completions(dataset=subject_dataset, prompter=get_subsubjects)
subsubject_dataset = []
for subject, subsubjects in zip(subject_dataset, result):
    subsubject_dataset.extend(
        [
            {"subject": subject.subject, "subsubject": subsubject.subject}
            for subsubject in subsubjects.subjects
        ]
    )

result = bella.completions(dataset=subsubject_dataset, prompter=get_qas)
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
