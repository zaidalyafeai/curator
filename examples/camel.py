from typing import Any, Dict, List

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
    user_prompt="For the given subject {{ subject }}. Generate 3 diverse subsubjects. No explanation.",
    response_format=Subjects,
    model_name="gpt-4o-mini",
)

GetQAList = prompt.Prompter(
    system_prompt="You are a helpful AI assistant.",
    user_prompt="For the given subject {{ subsubject }}, generate 10 diverse questions and answers. No explanation.",
    response_format=QAs,
    model_name="gpt-4o-mini",
)


def join_subject_subsubject(subject: Subject, subsubjects: Subjects) -> List[Dict[str, Any]]:
    return [
        {"subject": subject["subject"], "subsubject": subsubject["subject"]}
        for subsubject in subsubjects["subjects"]
    ]

def join_subsubject_qas(subsubject: Subject, qas: QAs) -> List[Dict[str, Any]]:
    return [
        {"subject": subsubject["subject"], "subsubject": subsubject["subsubject"], "question": qa["question"], "answer": qa["answer"]}
        for qa in qas["qas"]
    ]

subject_dataset = bella.completions(
    (),
    prompter=GetSubjects,
)
rows = []
for subject in subject_dataset:
    rows.extend(subject["subjects"])
subject_dataset = rows 


subsubjects_dataset = bella.completions(
    dataset=subject_dataset,
    prompter=GetSubSubjects,
)
rows = []
for subject, subsubjects in zip(subject_dataset, subsubjects_dataset):
    rows.extend(join_subject_subsubject(subject, subsubjects))
subsubject_dataset = rows 

print(pd.DataFrame.from_records(subsubject_dataset))

qa_dataset = bella.completions(
    dataset=subsubject_dataset,
    prompter=GetQAList,
)
rows = []
for subsubject, qas in zip(subsubject_dataset, qa_dataset):
    rows.extend(join_subsubject_qas(subsubject, qas))
qa_dataset = rows 

print(pd.DataFrame.from_records(qa_dataset))
