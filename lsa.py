#unsupervised learning method
#it extracts semantically significant sentences by applying SVD
#to the term-document frequency

import PyPDF2
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
import re
import string
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser

#creating a pdf file object
pdf = open(".\\abstract.pdf","rb")

#creating a pdf reader object
pdf_reader = PyPDF2.PdfReader(pdf)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#function for tokenization and removing stopwords
def remove_stop_words(text):
    punct_removed_text = text.translate(str.maketrans('','',string.punctuation))
    words = nltk.word_tokenize(punct_removed_text)
    
    #stop_words = set(stop_words.words('english'))
    
    words = [word for word in words if word.lower() not in stop_words]
    return " ".join(words)

#function for stemming
def stemming(text):
    tokens = text.split(' ')

    #defining a Stemmer
    stemmer = PorterStemmer()

    #stem the tokens 
    stemmed_tokens = []


    for token in tokens:
        stemmed_token = stemmer.stem(token)
        stemmed_tokens.append(stemmed_token)

    #join the stemmed tokens back into a string
    stemmed_text = ' '.join(stemmed_tokens)

    return stemmed_text



for i in range(0,len(pdf_reader.pages)):
    page = pdf_reader.pages[i]
    extracted_text = page.extract_text() # extraction happens here
    text_without_stopwords = remove_stop_words(extracted_text) #removing stopwords (function called)
    #stemmed_tokens = [stemmer.stem(word) for word in text_without_stopwords]
    sentences = sent_tokenize(extracted_text)
    #sentences = [sentence for sentence in sentences if not any(word.lower() in stop_words for word in sentence.split())]
    
    parser = PlaintextParser.from_string('\n'.join(sentences),Tokenizer('english'))
    lsa_summarizer = LsaSummarizer()
    lsa_summary = lsa_summarizer(parser.document,3)

    print("Summary : ", lsa_summary)
