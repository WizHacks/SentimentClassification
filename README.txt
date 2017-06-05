NAME
    sentiment_classification.py - Run the sentiment classification program

SYNOPSIS
    sentiment_classification.py [--epochs|--binary|--bagofwords|--tfidf|--nopunct|--nostopwords|--knn|--k|--metric|--rocchio|--p|--logistic_regression] [dataset]

DESCRIPTION
    Run the sentiment classification program with given preprocessing, vector representation model, classifier, and classifier options on the Movie Review Dataset.
    
    The dataset used in this program is the v2.0 polarity dataset from Cornell. There are two 
    classes in this dataset, positive and negative, each classes has 1000 files for a total of
    2000 files. If you would like to train on a similar dataset, please make sure that the 
    dataset follows this format:

    txt_sentoken/  
        positive_folder/
            file_1.txt file_2.txt ... file_42.txt
        negative_folder/
            file_43.txt file_44.txt ...

    Options:
        --epochs[=INT]
            how many times to iterate over training data
        --binary 
            use the binary feature vector model
        --bagofwords 
            use the bagofwords feature vector model
        --tfidf 
            use the tfidf feature vector model
        --nopunct 
            remove the punctation from the data
        --nostopwords
            remove stopwords from the data
        --knn 
            use the KNN classifier
        --k[=INT] 
            the k parameter fot the KNNclassifier. INT must be an integer value
        --metric[=DIS]
            use the specified distance metric when running classifier. DIS may be 'euclidean', 'manhattan' or 'minkowski'
        --rocchio
            use the nearest centroid classifier
        --p[=INT]
            use the p norm for when metric is minkowski. INT must be an integer value
        --logistic_regression
            use the logistic regression classifier


    If none of the optional arguments are provided, the program will be run with default options of not stripping the punctuation, epoch of 1, using a binary feature vector model, KNN classifier with k=1 and the Euclidean distance metric.

AUTHOR
    Wendy Zheng

COPYRIGHT
    Copyright (C) 2017 
