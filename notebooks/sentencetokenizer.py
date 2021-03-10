import re
import unicodedata

import pandas as pd
import spacy


def text_batches():
    # create list with 1000000 batch size. Spacy doc charecters limit
    file_open = open('E:/Sharpest_Mind/Weekly_Coding_Challenge/WikipediaCitationProject/all_text.txt', 'r', encoding='UTF-8',errors = 'ignore' )
    scraped_text = file_open.read()
    re.sub(r"\n", " ", scraped_text)   # remove newline charecter
    unicodedata.normalize("NFKD",scraped_text)   # encoding special charectersl
    file_open.close()
    total_count = len(scraped_text)
    chunks = (total_count - 1) // 1000000 + 1
    text_final =[]
    for i in range(chunks):
        batch = scraped_text[i*1000000:(i+1)*1000000]
        text_final.append(batch)
    return text_final


# sentence tokenization



def custom_sentence_boundary(doc):
    #  function to split sentences at the end of citation bracket and no splitting at some other charecters
    for i, token in enumerate(doc):
        if token.text == ']':
            doc[i + 1].sent_start = True
    return doc


def sentence_tokenization(text_batches):
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(custom_sentence_boundary, before='parser')
    sents_list = []
    for index, elem in enumerate(text_batches):
        doc = nlp(text_batches[index])
        for sent in doc.sents:
            sents_list.append(sent.text)
    df = pd.DataFrame(sents_list, columns=['text'])
    df.to_csv('sents_list.csv', index=False)
    return sents_list
