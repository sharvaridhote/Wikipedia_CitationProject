# Core Pkgs
# example problem from spacy-stremlit and modified by author
import streamlit as st
# NLP Pkgs
import spacy_streamlit
import spacy
#from text_explainer import explain_text_features

#nlp = spacy.load('en')
import os
from PIL import Image


def load_model():
    # declare global variables
    global nlp
    global textcat


nlp = spacy.load('model_artifactnWikidatatest1L2-2e-5')  ## will load the model from the model_path
textcat = nlp.get_pipe('textcat')  ## will load the model file


def predict(test_text):
    loaded_model1 = spacy.load('E:/Sharpest_Mind/WikipediaCitation/src/Notebooks/model_artifactnewdatatest1-2e-3D')
    textcat1 = loaded_model1.get_pipe('textcat')
    print("Sentence = ", test_text)  ## tweet
    txt_docs = list(loaded_model1.pipe(test_text))
    scores, _ = textcat1.predict(txt_docs)
    predicted_classes = scores.argmax(axis=1)
    result = ['Need citation' if lbl == 1 else 'Citation Not Required' for lbl in predicted_classes]

    return result[1]

def main():
    """A Simple NLP app with Spacy-Streamlit"""
    st.title("Wikipedia Citation Needed Predictor")
    our_image = Image.open(os.path.join('wiki_logo.png'))
    st.image(our_image)
    menu = ["Home", "NER", "Classification","Explain Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Tokenization")
        raw_text = st.text_area("Your Text", "Enter Text Here")
        docx = nlp(raw_text)
        if st.button("Tokenize"):
            spacy_streamlit.visualize_tokens(docx, attrs=['text', 'pos_', 'dep_', 'ent_type_'])
    elif choice == "NER":
        st.subheader("Named Entity Recognition")
        raw_text = st.text_area("Your Text", "Enter Text Here")
        docx = nlp(raw_text)
        spacy_streamlit.visualize_ner(docx, labels=nlp.get_pipe('ner').labels)
    elif choice == "Classification":
        st.subheader("Citation Needed ")
        raw_text = st.text_area("Your Text", "Enter Text Here")
        docx = nlp(raw_text)
        spacy_streamlit.visualize_textcat(docx, title="Sentence Need Citation" )
    elif choice == "Explain Prediction":
        st.subheader("Why this predicted")
        raw_text = st.text_area("Your Text", "Enter Text Here")
        # get number of features input
        num_features_input = st.number_input(label="Num of features to visualize",
                                             min_value=1, max_value=7, step=1)



if __name__ == '__main__':
    main()
