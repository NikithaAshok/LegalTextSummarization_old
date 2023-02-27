#T5 is an encoder-decoder model. Converts all language problems into text-to-text format.

from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import PyPDF2
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

#creating a pdf file object
pdf = open(".\\abstract.pdf","rb")

#creating a pdf reader object
pdf_reader = PyPDF2.PdfReader(pdf)

#checking number of pages in a pdf file
#print("Number of pages in the pdf ",len(pdf_reader.pages))

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


#stemming tokens 

#function for tokenization and removing stopwords
def remove_stop_words(text):
    punct_removed_text = text.translate(str.maketrans('','',string.punctuation))
    words = nltk.word_tokenize(punct_removed_text)
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

#instantiating the model and tokenizer
my_model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')


for i in range(0,len(pdf_reader.pages)):
    page = pdf_reader.pages[i]
    extracted_text = page.extract_text() # extraction happens here
    text_without_stopwords = remove_stop_words(extracted_text) #removing stopwords (function called)
    sentences = sent_tokenize(extracted_text)
    text = "summarize:" + sentences

    #T5 MODEL -----------
    #converting input sequence to input-ids through process of encoding - encode()
    input_ids = tokenizer.encode(text,return_tensors='pt',max_length=512)
    #generate() method returns a sequence of ids corresponding to the summary
    summary_ids = my_model.generate(input_ids)
    #using decode() function to generate summary text from the above ids
    t5_summary = tokenizer.decode(summary_ids[0])
    print(t5_summary)
