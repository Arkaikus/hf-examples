""" 
usage examples of transformers.pipeline 

no need of any model download, it will download the model automatically
after first run of the pipeline models will be cached in ~/.cache/huggingface/hub
"""

from transformers import pipeline

print("Sentiment Analysis with distilbert model")
sentiment_classifier = pipeline("sentiment-analysis")  # will use default distilbert model
result = sentiment_classifier("We are very happy to introduce pipeline to the transformers repository.")
print(result)  # should print positive

result = sentiment_classifier("We are devastated you want to uninstall transformers repository.")
print(result)  # should print negative

result = sentiment_classifier("Estamos muy felices de presentar el modulo pipeline en el repositorio transformers.")
print(result)  # wrong language, should print negative

result = sentiment_classifier("Estamos devastados de que quieras desinstalar el repositorio de transformers.")
print(result)  # wrong language, should print negative

print("\nSentiment Analysis with bert-base-multilingual-uncased-sentiment model")
# Classify sentiment in multiple languages from 5 stars to 1 star
sentiment_classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
result = sentiment_classifier("We are very happy to introduce pipeline to the transformers repository.")
print(result)  # should print 5 stars

result = sentiment_classifier("We are devastated you want to uninstall transformers repository.")
print(result)  # should print 1 star

result = sentiment_classifier("Estamos muy felices de presentar el modulo pipeline en el repositorio transformers.")
print(result)  # should print 5 stars

result = sentiment_classifier("Estamos devastados de que quieras desinstalar el repositorio de transformers.")
print(result)  # should print 1 star
