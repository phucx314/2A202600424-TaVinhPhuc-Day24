
import os
from dotenv import load_dotenv
load_dotenv()
from ragas import evaluate
from ragas.metrics import faithfulness
from ragas.llms import llm_factory
from datasets import Dataset

data = {
    "user_input": ["What is RAG?", "Who is the CEO of Apple?"],
    "response": ["RAG is retrieval augmented generation.", "Tim Cook is the CEO of Apple."],
    "retrieved_contexts": [["RAG stands for Retrieval-Augmented Generation."], ["Tim Cook became CEO in 2011."]],
    "reference": ["RAG is a technique to ground LLMs.", "The CEO of Apple is Tim Cook."]
}
dataset = Dataset.from_dict(data)

print("Starting debug evaluation with default LLM...")
results = evaluate(dataset, metrics=[faithfulness])
print("Results:", results)
