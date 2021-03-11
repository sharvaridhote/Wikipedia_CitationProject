import random
import re
import unicodedata
import sklearn.utils


def text_cleaning(clean_text):
    """
    text cleaning to remove any html tags, or unexpected charecters during encodings
    while reading text files or dataframe
    parameter:
    clean_text: str - input text
    return: str - cleaned text
    """
    clean_text = clean_text.str.replace("(<br/>)", "")
    clean_text = clean_text.str.replace('(<a).*(>).*(</a>)', '')
    clean_text = clean_text.str.replace('(&amp)', '')
    clean_text = clean_text.str.replace('(&gt)', '')
    clean_text = clean_text.str.replace('(&lt)', '')
    clean_text = clean_text.str.replace('(\xa0)', ' ')
    return clean_text


def create_label(df):
    """ Create labels and removes citation brackets from the text
        df : pandas dataframe with text column
        return: cleaned pandas dataframe with labels
        """
    df = sklearn.utils.shuffle(df)
    for col in df.columns:
        df[col], df['label'] = zip(*df[col].apply(lambda x: label_creator(x)))
    return df


def string_limit(df):
    """ It helps to remove the short sentences, words and charecters which
    are miss-split during Spacy tokenization
    parameters:
    df : pandas dataframe with text column
    return: cleaned pandas dataframe
    """
    for col in df.columns:
        df = df[(df[col].str.split().str.len() > 30)].reset_index(drop=True)
    return df


def unicode_text(df):
    """
    Text preprocessing of Charecter encoding of text columns
    parameter:
    df : pandas dataframe
    return: pandas dataframe

    """
    for col in df.columns:
        df[col] = (df[col].map(lambda x: unicodedata.normalize('NFKD', str(x))))
    return df


def label_cat(df):
    """
    Convert labeled train data to spacy format POSITIVE - TRUE and NEGATIVE : FALSE
    df : pandas dataframe of text and labels in a tuple
    return: labelled dataframe
    """
    # shuffle
    df = df_tolist(df)
    df = sklearn.utils.shuffle(df)
    df_texts, df_cats = zip(*df)
    # get the categories for each sentence
    test_cats = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} for y in df_cats]
    df_test = list(zip(df_texts, [{'cats': cats} for cats in df_cats]))
    return df_test, df_texts, df_cats


def df_tolist(df):
    df[df.columns[0]] = (df[df.columns[0]].map(lambda x: unicodedata.normalize('NFKD', str(x))))
    df = sklearn.utils.shuffle(df)
    df['tuples'] = df.apply(lambda row: (row[df.columns[0]], row[df.columns[1]]), axis=1)
    df = df['tuples'].tolist()
    return df


def label_creator(x):
    """
    Find and remove citation from the text and creates labels
    parameters:
        x : str - charecters in the string
    returns :
        cleanx : str - cleaned text without citation
        label : int - sentence with citation: 1 else 0
    """
    # suffix = re.compile("\[(.*?)\]$")
    infix = re.compile('\[(.+?)\]')
    clean_x = re.findall(infix, x)

    if len(clean_x):
        label = 1
    else:
        label = 0
    cleanx = re.sub(infix, '', x)
    return cleanx, label






