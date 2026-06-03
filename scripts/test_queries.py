import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_core import answer_question


QUESTIONS = [
    "Find an article that reframes marketing as a conversation with readers, aimed at writers who find self-promotion uncomfortable. Provide the title and author.",
    "List exactly 3 articles about education. Return only the titles.",
    "Find an article that argues past pandemics (such as the bubonic plague) can spur innovation and recovery, and summarise its central argument.",
    "I want practical, beginner-friendly advice on building habits that actually stick. Which article would you recommend, and why?",
]


for question in QUESTIONS:
    print("=" * 80)
    print(question)
    result = answer_question(question)
    print(result["response"])
    print("Context titles:")
    for item in result["context"]:
        print(f"- {item['score']:.4f} | {item['title']} | {item['article_id']}")
