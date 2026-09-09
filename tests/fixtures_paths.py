import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINI_Q = os.path.join(_ROOT, "tests", "data", "pubmed_qa_mini", "questions.json")
MINI_C = os.path.join(_ROOT, "tests", "data", "pubmed_qa_mini", "corpus.json")
E2E_MINI = os.path.join(_ROOT, "tests", "data", "configs", "e2e_mini.yaml")
