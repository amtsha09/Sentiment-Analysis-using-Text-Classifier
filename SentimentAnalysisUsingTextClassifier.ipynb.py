# Sentiment-Analysis-using-Text-Classifier

from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import urllib.request

def read_data(path):
    """
    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])

def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. 
    The token "isn't" is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.

    """

    doc = doc.lower()
    if keep_internal_punct == False:
        doc = re.sub("[\W]+", " ", doc).split()
    elif keep_internal_punct == True:
        doc = [word.strip(string.punctuation) for word in doc.split()]
    return np.array(doc)

def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    
    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    """
    for token in tokens:
        feats['token=' + token] = feats['token=' + token] + 1

def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.

    """
    for i in range(len(tokens) - k + 1):
        window_pair = list(combinations(tokens[i:i + k], 2))

        for wp in window_pair:
            feats['token_pair=' + wp[0] + '__' + wp[1]] = feats['token_pair=' + wp[0] + '__' + wp[1]] + 1

neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    """
    feats['neg_words'] = 0
    feats['pos_words'] = 0

    for i in map(lambda x: x.lower(), tokens):
        if i in neg_words:
            feats['neg_words'] = feats['neg_words'] + 1

        elif i in pos_words:
            feats['pos_words'] = feats['pos_words'] + 1

def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.

    """
    feats = defaultdict(lambda: 0)
    for feature in feature_fns:
        feature(tokens, feats)
    return sorted(feats.items(), key=lambda f: f[0])

def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. 
      NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically)

    """

    row = list()
    v = list()
    column = list()
    data = list()
    complete_feat = list()
    cnt = Counter()

    for token in tokens_list:
        feats = featurize(token, feature_fns)
        complete_feat += feats

    for d in complete_feat:
        if d[0] == "neg_words" and d[1] < 1 or d[0] == "pos_words" and d[1] < 1:
            continue
        else:
            cnt[d[0]] += 1

    for i, j in cnt.items():
        if j >= min_freq:
            v.append(i)
    v.sort()


    if vocab == None:
        vocab = defaultdict(lambda: 0)
        for index, value in enumerate(v):
            vocab[value] = index

    for index, token in enumerate(tokens_list):
        feats = featurize(token, feature_fns)
        for f in feats:
            if f[0] in vocab:
                data.append(f[1])
                row.append(index)
                column.append(vocab[f[0]])

    row = np.array(row)
    column = np.array(column)
    data = np.array(data)
    
    return (csr_matrix((data, (row, column)), shape=(len(tokens_list), len(vocab)), dtype=np.int64), vocab)

def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. 

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    cv = KFold(len(labels), k)
    accuracies = []

    for train_idx, test_idx in cv:
        clf.fit(X[train_idx], labels[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(labels[test_idx], predicted)
        accuracies.append(acc)
    avg = np.mean(accuracies)
    return avg

def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.

    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.

      This list should be SORTED in descending order of accuracy.

    """
    clf = LogisticRegression()
    eac_list = list()

    combi = list()
    for i in range(1, len(feature_fns) + 1):
        combi += list(combinations(feature_fns, i))

    for feature in combi:
        for pv in punct_vals:
            listOfTokens = [tokenize(doc, keep_internal_punct=pv) for doc in docs]
            for mf in min_freqs:
                X, vocab = vectorize(listOfTokens, feature, min_freq=mf)
                accuracy = cross_validation_accuracy(clf, X, labels, 5)
                eac_dict = dict()
                eac_dict['punct'] = pv
                eac_dict['features'] = feature
                eac_dict['min_freq'] = mf
                eac_dict['accuracy'] = accuracy

                eac_list.append(eac_dict)

    newlist = sorted(eac_list, key=lambda z: -z['accuracy'])

    return(newlist)

def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    li = list()
    for l in results:
        li.append(l['accuracy'])
    li.reverse()
    plt.plot(li)
    plt.ylabel('ACCURACY')
    plt.xlabel('SETTING')
    # plt.show()
    plt.savefig("accuracies.png")

def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. 

    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    fields = list()
    for result in results:
        fields.append(("min_freq", result['min_freq']))
        fields.append(("punct", result['punct']))
        fields.append(("features", result['features']))

    fields = set(fields)
    accuracy_setting = list()

    for field in fields:
        mean_accu_list = list()
        for result in results:
            if result[field[0]] == field[1]:
                mean_accu_list.append(result['accuracy'])
        if field[0] == "features":
            setting = field[0] + "=" + " ".join([i.__name__ for i in field[1]])
        else:
            setting = field[0] + "=" + str(field[1])

        mean_accu = sum(mean_accu_list) / len(mean_accu_list)
        accuracy_setting.append((mean_accu, setting))

    return (sorted(accuracy_setting, key=lambda x: -x[0]))


def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.

    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    clf = LogisticRegression()

    listOfTokens = [tokenize(doc, keep_internal_punct=best_result['punct']) for doc in docs]
    X, vocab = vectorize(listOfTokens, best_result['features'], min_freq=best_result['min_freq'])
    clf.fit(X, labels)

    return (clf, vocab)


def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.

    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
    coeff = clf.coef_[0]
    sorted_vocab = sorted(vocab.keys())
    zipped = zip(sorted_vocab, coeff)

    pos_feature_name = list()
    neg_feature_name = list()

    for z in zipped:
        if z[1] >= 0:
            pos_feature_name.append(z)
        elif z[1] < 0:
            p = (z[0],abs(z[1]))
            neg_feature_name.append(p)

    pos = sorted(pos_feature_name, key=lambda x: -x[1])
    neg = sorted(neg_feature_name, key=lambda x: -x[1])

    if label == 1:
        return pos[:n]
    elif label == 0:
        return neg[:n]

def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.


    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """

    test_docs, test_labels = read_data(os.path.join('data', 'test'))
    tok = [tokenize(token, best_result["punct"]) for token in test_docs]

    X_test, vocab = vectorize(tok, best_result["features"], best_result["min_freq"], vocab)

    return (test_docs, test_labels, X_test)

def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. 

    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.

    Returns:
      Nothing
    """
    predict_value = clf.predict(X_test)
    #     print(predict_value)
    predict_proba = clf.predict_proba(X_test)
    #     print(predict_proba)
    index_proba = defaultdict()

    for ind, val in enumerate(predict_value):
        if val != test_labels[ind]:
            index_proba[ind] = predict_proba[ind][val]
            #     print(index_proba)

    index_proba_sorted = sorted(index_proba.items(), key=lambda x: -x[1])
    #     print(index_proba_sorted)

    li = list()
    for i in range(n):
        li.append(index_proba_sorted[i])
    # print(li)

    for i in li:
        print("truth", "=", test_labels[i[0]], " ", "predicted", "=", predict_value[i[0]], " ", "proba", "=", str(i[1]))
        print(test_docs[i[0]])
        print("\n")


def main():
 
    feature_fns = [token_features, token_pair_features, lexicon_features]
    
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))

    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()










