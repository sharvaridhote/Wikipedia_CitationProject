import random
import numpy as np
import spacy
from spacy.util import minibatch, compounding, decaying
from util import df_tolist
from pathlib import Path

def load_data(df, split=0.2):
    """
    Prepare the training data as per Spacy format
    Parameters:
        df: training data in pandas dataframe
        split: float - Splitting dataframe to train and validation set. Defaults to 0.2
    Returns:
        tuples: train and validation text and labels
    """
    # Shuffle the data
    df_train = df_tolist(df)
    random.shuffle(df_train)
    texts, labels = zip(*df_train)
    # get the categories for each sentence
    cats = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} for y in labels]
    # Splitting the training and evaluation data
    split = int(len(df_train) * split)
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])


def evaluate(tokenizer, textcat, texts, cats):
    """
    Evaluate the performance of TextCategoriser prediction
    Calculate accuracy, f1 score, precision, recall
    parameters:
        nlp: object - spacy
        textcat: TextCategoriser
        texts : input text to be evaluated
        cats : input label
    """
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives

    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label == "NEGATIVE":
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 > gold[label]:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 <= gold[label]:
                fn += 1
    # calculate metrics
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)

    return {"textcat_a": accuracy, "textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}
def training(train_texts, train_cats, dev_texts, dev_cats, test_texts, test_cats, L2, learn_rate, n_iter,  output_dir=None):
    """
       Spacy example function modified
       Trains citation needed classifier and saves model
       Parameters:
           train_texts :str -list - text train features
           train_cats :str - list - label citation sentence - TRUE else FALSE
           dev_texts :str - list - text train features
           dev_cats :str - list - label citation sentence - TRUE else FALSE
           test_texts :str - list - text train features
           test_cats :str - list - label citation sentence - TRUE else FALSE
           L2 : int - regularization parameter - default value 1e-6
           learn_rate : learning rate - default rate - 0.001,
           output_dir :str = None - path to save the model
       returns:
           returns list of evaluated metrics (accuracy, f1, precision and recall)
           train_results : list - evaluated metrics for training dataset
           val_results : list - evaluated metrics for validation dataset
       """

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

    # Disabling other components
    nlp = spacy.load('en_core_web_sm')
    # Adding the built-in textcat component to the pipeline.
    textcat = nlp.create_pipe("textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"})
    nlp.add_pipe(textcat, last=True)
    # Adding the labels to textcat
    textcat.add_label("POSITIVE")
    textcat.add_label("NEGATIVE")
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        optimizer.L2 = L2
        optimizer.learn_rate = learn_rate
        #dec = decaying(0.6, 0.2, 1e-4)
        dec = decaying(10.0, 1.0, 0.001)
        print("Training the model...")
        print('{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'A_train', 'A_dev', 'A_test', 'P', 'R', 'F'))
        train_results = []
        dev_results = []
        test_results = []
        # Performing training
        for i in range(n_iter):
            losses = {}
            train_data = list(zip(train_texts, [{'cats': cats} for cats in train_cats]))
            random.shuffle(train_data)
            # (train_texts, train_cats) = zip(*train_data)
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=next(dec),
                           losses=losses)
            # Calling the evaluate() function and printing the train scores
            scores1 = evaluate(nlp.tokenizer, textcat, train_texts, train_cats)
            train_results.append(scores1)
            # Calling the evaluate() function and printing the test scores
            with textcat.model.use_params(optimizer.averages):

                scores2 = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
                scores3 = evaluate(nlp.tokenizer, textcat, test_texts, test_cats)
            dev_results.append(scores2)
            test_results.append(scores3)
            print('{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}'
                  .format(losses['textcat'], scores1['textcat_a'], scores2['textcat_a'], scores3['textcat_a'],
                          scores1['textcat_p'],
                          scores1['textcat_r'], scores1['textcat_f']))
    if output_dir is not None:
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

    return train_results, dev_results, test_results
