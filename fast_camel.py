import bella
import time
import litellm
import dask

from pydantic import BaseModel, Field

litellm.set_verbose = False


class Subjects(BaseModel):
  subjects: list[str] = Field(description="A list of subjects")

class QA(BaseModel):
    question: str = Field(description="A question")
    answer: str = Field(description="A answer")

class QAList(BaseModel):
    qa: list[QA] = Field(description="A list of questions and answers")

PromptLayer = bella.PromptLayer
DelayedPromptLayer = bella.DelayedPromptLayer
Models = bella.Models


def camel():
  GetSubjects = DelayedPromptLayer(
    system_prompt="Generate a diverse list of 10 subjects. Keep it high-level (e.g. Math, Science)",
    response_format=Subjects, model_name=Models.GPT_4O)


  GetSubSubjects = DelayedPromptLayer(
    system_prompt="For the given subject, generate 3 diverse sub-subjects.",
    response_format=Subjects, model_name=Models.GPT_4O)

  QAPrompt = DelayedPromptLayer(
    system_prompt="For the given subject, generate 3 diverse questions and answers.",
    response_format=QAList, model_name=Models.GPT_4O_MINI)

  subjects = GetSubjects()  
  d = {}
  for i in range(10):
    l = []
    sub_subjects = GetSubSubjects(subjects.subjects[i])
    for j in range(3):
      sub_subject = sub_subjects.subjects[j]
      qa_list = QAPrompt(sub_subject)
      for k in range(3):
        l.append(qa_list.qa[k])
    d[subjects.subjects[i]] = l

  return dask.compute(d)[0]


def main():
  start = time.time()
  d = camel()
  print(d)
  print(f"Time taken: {time.time() - start}")


if __name__ == "__main__":
  main()