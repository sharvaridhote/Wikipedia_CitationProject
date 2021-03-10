#import libraries
#import GPUtil
#import torch
from train import training



#myurl = get_url()
#all_pages = get_weblinks(myurl)
#web_links = extract_weblinks(myurl, all_pages)
#all_text = get_text(web_links)
#sents_token = sentence_tokenization()

# Get train and dev data cleaned, creat label and change to spacy input data format
# df_train = pd.read_csv('E:/Sharpest_Mind/WikipediaCitation/notebooks/Final_cleaned5K.csv', encoding = 'ISO-8859-1')
# df_train['text_new'] = (df_train['text_new'].map(lambda x: unicodedata.normalize('NFKD', str(x))))
# df_train['cleantext'], df_train['label'] = zip(*df_train['text_new'].apply(lambda x: labelcreator(x)))
# df_train['text_edit'] = df_train['cleantext'].str.replace(r'\[(.+?)\]', '')
# df_train['tuples'] = df_train.apply(lambda row: (row['text_edit'], row['label']), axis=1)
# df_train = df_train['tuples'].tolist()
#
# # test data cleaned and, creat label change to spacy input data format
# df_test = pd.read_csv('E:/Sharpest_Mind/WikipediaCitation/notebooks/Cleaned_test_3Ktdata.csv', encoding = 'utf8')
# df_test['text_new'] = (df_test['text_new'].map(lambda x: unicodedata.normalize('NFKD', str(x))))
# df_test['cleantext'], df_test['label'] = zip(*df_test['text_new'].apply(lambda x: labelcreator(x)))
# df_test['text_edit'] = df_test['cleantext'].str.replace(r'\[(.+?)\]', '')
# df_test['tuples'] = df_test.apply(lambda row: (row['text_edit'], row['label']), axis=1)
# df_test = df_test['tuples'].tolist()


# Model 1 - Ensemble, L2 =2e-4
train_results1, dev_results1, test_results1 = training(train_texts, train_cats, dev_texts, dev_cats, test_texts1, test_cats1, L2 = 2e-5,
                                                     learn_rate = 0.001, n_iter = 3, output_dir='model_artifactnnewdata')