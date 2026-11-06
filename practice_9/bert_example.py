from transformers import pipeline

nlp = pipeline("fill-mask", model="bert-base-uncased")

result = nlp("Deep learning is the [MASK] of NLP.")
print(result)
