"""
===============================================================================
Sentiment Classification of text documents using the Perceptron and Naive Bayes Model
===============================================================================

This is a program that will go through the stages of preprocessing data, training the
data using KNN or Nearest Centroid classification, and testing the models using five-fold cross 
validation techniques. A preprocessing option to strip punctionuation, and (possibly) stop words 
can also be specified. The features vectors of the data can be represented in either binary, bag of words, 
or (maybe) tf-idf approach. These options can be configured by passing the respective parameters as 
arguments. 

The default configuration will be binary model with the punctuation in tact, and it will be trained 
on a KNN classifier with k = 1 and using euclidean distance metrics. 

The dataset used in this program is the v2.0 polarity dataset from Cornell. There are two 
classes in this dataset, positive and negative, each classes has 1000 files for a total of
2000 files. If you would like to train on a similar dataset, please make sure that the 
dataset follows this format:

txt_sentoken/  
    positive_folder/
        file_1.txt file_2.txt ... file_42.txt
    negative_folder/
        file_43.txt file_44.txt ...
        
"""

import os
import numpy as np
import process_data
import time
import distance_metrics
from sys import exit
from process_data import BinaryModel, BagOfWordsModel, TFIDFModel
from knn import KNN
from rocchio import Rocchio

from argparse import ArgumentParser, ArgumentTypeError
from sklearn.datasets import load_files
from sklearn.model_selection import KFold
   
# parse commandline arguments
usage = "%(prog)s [options] dataset"
ap = ArgumentParser(usage = usage)

# Text Preprocessing
ap.add_argument("--nopunct",
                action="store_true",
                default=False,
                help="Remove the punctuation from the data.")

ap.add_argument("--nostopwords",
                action="store_true",
                default=False,
                help="Remove the stop words from the data.")

# Feature Vector Modeling
ap.add_argument("--binary", 
                action="store_true",
                default=False,
                help="Represent feature vector using binary model.")

ap.add_argument("--bagofwords", 
                action="store_true",
                default=False,
                help="Represent feature vector using bag of words model.")

ap.add_argument("--tfidf", 
                action="store_true",
                default = False,
                help="Represent feature vector using tfidf model.")

# Classifiers
ap.add_argument("--knn", 
                action="store_true",
                default=False,
                help="Train the data using KNN classifier.")  

ap.add_argument("--k", 
                action="store",
                default=1,
                help="Train the data using KNN classifier with given k. Must be an integer, otherwise ignored.")

ap.add_argument("--metric", 
                action="store",
                default="euclidean",
                help="Train the data using KNN classifier with given distance metric.")

ap.add_argument("--p", 
                action="store",
                default=1,
                help="Used the p norm specified for minkowski distance.")

ap.add_argument("--rocchio", 
                action="store_true",
                default=False,
                help="Train the data using nearest centroid classifier.")  

# Required
ap.add_argument("dataset",
                help="The directory of the dataset.")
args = ap.parse_args()

# Error Checking
if args.knn and args.rocchio:
    print "Please check your arguments!"
    ap.print_help()
    exit(1)

if (args.binary and args.bagofwords) or (args.binary and args.tfidf) or (args.bagofwords and args.tfidf):
    print "Please check your arguments!"
    ap.print_help()
    exit(1)

# load the data from the dataset shuffle it randomly
# a seed is used for testing purposes
if os.path.isdir(args.dataset):
    print "Loading dataset from:", args.dataset
    data = load_files(container_path=args.dataset, load_content=True, shuffle=True)
    #data = load_files(container_path=args.dataset, load_content=True, shuffle=True, random_state=None)
    print "Loaded: %d files" %(len(data.filenames))

    # Convert to numpy array for easy processing
    data.data = np.array(data.data)
    data.target = np.array(data.target)

    # Create the 5 fold cross validation sets
    folds = 5
    iteration = 1
    results = []
    kf = KFold(n_splits=folds)
    for train_index, test_index in kf.split(data.data):        
        #print "TRAIN:", train_index, "TEST:", test_index
        print "Cross validating on fold %d" %(iteration%(folds+1))
        start = time.time()    
        iteration+=1
        # Process the data before creating feature vector
        # Strip the punctuation if specified
        if args.nopunct:        
            data.data[train_index] = process_data.strip_punct(data.data[train_index])
            data.data[test_index] = process_data.strip_punct(data.data[test_index])
        # Remove the stop words if specified    
        if args.nostopwords:            
            data.data[train_index] = process_data.remove_stopwords(data.data[train_index]) 
            data.data[test_index] = process_data.remove_stopwords(data.data[test_index])   
                
        # Find the vocabulary of the dataset
        train_vocabulary = process_data.get_vocabulary(data.data[train_index])        

        # Start modeling the text        
        # Bag of Words Model    
        if args.bagofwords:
            bow_model = BagOfWordsModel(train_vocabulary)
            data.feature_vectors = bow_model.create_bow_model(data.data)        

        # TF-IDF Model
        elif args.tfidf:
            tfidf_model = TFIDFModel(train_vocabulary)
            data.feature_vectors = tfidf_model.create_tfidf_model(data.data)    
            # Process the feature vectors
            [process_data.normalize(x) for x in data.feature_vectors]      

        # Binary/Boolean Model
        else:
            binary_model = BinaryModel(train_vocabulary)
            data.feature_vectors = binary_model.create_binary_model(data.data)       
        
        # Classifiers
        # Nearest Centroid
        if args.rocchio:            
            if args.metric == "manhattan":  
                print "manhattan"               
                classifier = Rocchio(data.target_names, distance_metrics.manhattan_scipy)
            elif args.metric == "minkowski":
                try:
                    args.p = int(args.p)
                    print "minkowski p=%d" %args.p
                    classifier = Rocchio(data.target_names, distance_metrics.minkowski_scipy, args.p)
                except ValueError:
                    print "Please check your arguments!"
                    ap.print_help()
                    exit(1)
            else:
                print "euclidian"
                classifier = Rocchio(data.target_names, distance_metrics.euclidean_scipy)
        # KNN 
        else:
            try:
                args.k = int(args.k)
                if args.metric == "manhattan":   
                    print "manhattan"            
                    classifier = KNN(distance_metrics.manhattan_scipy, args.k)
                elif args.metric == "minkowski":
                    args.p = int(args.p)
                    print "minkowski p=%d" %args.p
                    classifier = KNN(distance_metrics.minkowski_scipy, args.k, args.p)
                else:
                    print "euclidian"
                    classifier = KNN(distance_metrics.euclidean_scipy, args.k)
            except ValueError:
                print "Please check your arguments!"
                ap.print_help()
                exit(1)
        
        # Train the classifer on the training data and test on seperate data
        classifier.train(data.feature_vectors[train_index], data.target[train_index])
        results.append(classifier.score(data.feature_vectors[test_index], data.target[test_index]))
        end = time.time()
        print "training and testing time:", end - start

    # Accuracy-fraction classified correctly
    # Precision-fraction of retrieved instances that are relevant
    # Recall-fraction of relevant instances that are retrieved   
    accuracy_avg = float(sum(acc[0] for acc in results))/folds
    precision_avg = float(sum(pre[1] for pre in results))/folds
    recall_avg = float(sum(re[2] for re in results))/folds
    print "avg accuracy:", accuracy_avg
    print "avg precision:", precision_avg 
    print "avg recall:", recall_avg       

else:
    print "Invalid directory:", args.dataset
    ap.print_help()


