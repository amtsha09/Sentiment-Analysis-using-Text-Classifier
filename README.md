# Sentiment-Analysis-using-Text-Classifier

Here we are building a text classifier to determine whether a movie review is expressing positive or negative sentiment. 
The movie reviews data we are using is from IMDB.COM

The classifier we are using is Logistic Regression Classifier along with 10-fold Cross validation.

We have used different ways to preprocess the data i.e. by creating different features. 

It involves 3 approaches:
#### 1. token features
    count of each token i.e. how many times a token occurs in a review
#### 2. token pair features
    count of each pair of token limited by window size. 
#### 3. lexicon features
    count how many times a token appears that matches either the neg_words or pos_words.

Then by comparing the cross-validation accuracy of each approach. We find which setting is best suited for classification.
Then, we computed accuracy on a test set and did some analysis of the errors.
