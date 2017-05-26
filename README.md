# Text Sentiment Classification

## System Dependencies
- Python 2.7
- NumPy
- Scipy

## The Program
This is a program that will go through the stages of preprocessing data, training the
data using perceptron, naive bayes, knn and rocchio classifiers and testing the models using five-fold cross validation techniques. A preprocessing option to strip punctionuation, and/or stop words can also be specified. The features vectors of the data can be represented in either binary, bag of words, or tf-idf approach. These options can be configured by passing the respective parameters as arguments. 

If none of the optional arguments are provided, the program will be run with default options of not stripping the punctuation, using a binary feature vector model, KNN classifier with k=1 and the Euclidean distance metric.

## The Dataset
The dataset used in this program is the [v2.0 polarity dataset](https://www.cs.cornell.edu/people/pabo/movie-review-data/) from Cornell. There are two classes in this dataset, positive and negative, each classes has 1000 files for a total of 2000 files. 
If you would like to train on a similar dataset, please make sure that the 
dataset follows this format:
```
txt_sentoken/  
    positive_folder/
        file_1.txt file_2.txt ... file_42.txt
    negative_folder/
        file_43.txt file_44.txt ...
```        

### Running the Program
A sample run of the program might be:
```
python sentiment_classification.py review_polarity/txt_sentoken --nopunct
```
This would run the program with the default perceptron classifier using the binary model to create feature vectors after stripping out the punctuation from the dataset with the euclidean distance metric.

### The Algorithms
- [KNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [Bag of Words Model](https://en.wikipedia.org/wiki/Bag-of-words_model)
- [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
- [Stopwords](http://www.ranks.nl/stopwords)
- [Binary Independence Model](https://en.wikipedia.org/wiki/Binary_Independence_Model)
- [Perceptron Binary Classifier](https://en.wikipedia.org/wiki/Perceptron)
- [Naive Bayes Probabilistic Model](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

### To Do:
- default number of stopwords is 173, which is a bit high and gives lower accuracy 
- try normalizing binary to see if better accuracy
- save and load data from csv files