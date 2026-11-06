from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

texts = [
    "I love this movie, it was amazing!",
    "This film was boring and too long.",
    "The acting was good but the plot was weak.",
]

for text, res in zip(texts, classifier(texts)):
    print(text, "->", res["label"], round(res["score"], 4))
