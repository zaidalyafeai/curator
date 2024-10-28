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

GetSubjects = PromptLayer(
  system_prompt="Generate a diverse list of 10 subjects. Keep it high-level (e.g. Math, Science)",
  response_format=Subjects, model_name=Models.GPT_4O)

GetSubSubjects = PromptLayer(
  system_prompt="For the given subject, generate 3 diverse sub-subjects.",
  response_format=Subjects, model_name=Models.GPT_4O)

QAPrompt = PromptLayer(
  system_prompt="For the given subject, generate 3 diverse questions and answers.",
  response_format=QAList, model_name=Models.GPT_4O_MINI)


def slow_camel():
  """All calls are sequential."""
  subjects = GetSubjects()  
  d = {}
  for subject in subjects.subjects:
    l = []
    for sub_subject in GetSubSubjects(subject).subjects:
      qa_list = QAPrompt(sub_subject)
      l.extend(qa_list.qa)
    d[subject] = l

  return d


def main():
  start = time.time()
  d = slow_camel()
  print(d)
  print(f"Time taken: {time.time() - start}")


if __name__ == "__main__":
  main()