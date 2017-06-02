import numpy as np

class LogisticRegression():
    '''LogisticRegression classifier'''
    def __init__(self, classes, learning_rate, epochs = 1, use_intercept=False):
        self.classes = classes
        self.learning_rate = learning_rate
        self.use_intercept = use_intercept
        self.epochs = epochs # how many times to go over training data
        print "Logistic Regression initialize"

    def train(self, X, y):
        '''Train the logistic regression classifier on training data with correctly labeled classes'''
        print "Training logistic regression classifer on %d documents ..." %(X.shape[0])
        # Add ones to the end of all the feature vectors
        '''
        >>> a = np.array([[1],[2],[3]])
        >>> b = np.array([[2],[3],[4]])
        >>> np.hstack((a,b))
        array([[1, 2],
               [2, 3],
               [3, 4]])
        '''
        if self.use_intercept:
            intercept = np.ones((X.shape[0]), 1)
            X = np.hstack((intercept, X))

        # length of feature vector weights
        weights = np.zeros(X.shape[1])    

        # how many iterations over the training data
        for epoch in self.epochs:
            for fvi in range(y.size):
                scores = np.dot(X, weights)
                predictions = sigmoid(scores)

    def score(self, X, y):
        '''Score the logistic regression classifier on test data'''
        print "Testing logistic regression classifer on %d documents ..." %(X.shape[0])
        tp = fp = tn = fn = 0
        for fvi in range(X.shape[0]):               
            d = y[fvi]
            y_hat = self.get_nearest_centroid(X[fvi])
            if d == 1:
                if y_hat == 1:
                    tp+=1
                else:
                    fp+=1
            elif d == 0:
                if y_hat == 0:
                    tn+=1
                else:
                    fn+=1
        print "tp", tp, "tn", tn, "fp", fp, "fn", fn   
        if tp == 0:
            p_precision = 1
            p_recall = 1
        else:
            p_precision = float(tp)/(tp+fp)   
            p_recall = float(tp)/(tp+fn)  
        if tn == 0:
            n_precision = 1          
            n_recall = 1
        else:
            n_precision = float(tn)/(tn+fn)            
            n_recall = float(tn)/(tn+fp)            
        accuracy = float((tp+tn))/(tp+tn+fp+fn)
        precision = float((p_precision + n_precision))/2   
        recall = float((p_recall + n_recall))/2              
        print "+precision %f" %(p_precision)
        print "-precision %f" %(n_precision)
        print "+recall %f" %(p_recall)
        print "-recall %f" %(n_recall)
        print "accuracy %f" %(accuracy)
        print "precision %f" %(precision)
        print "recall %f" %(recall)
        return (accuracy, precision, recall)

    def predict(self, X):
        '''Given a new set of test documents, predict their labels'''
        print "Predicting on %d documents ..." %(X.shape[0])
        predictions = [self.get_nearest_centroid(X[fvi]) for fvi in range(X.shape[0])]
        return np.array(predictions)
        
    def sigmoid(self, X):
        return 1/(1 + np.exp(-X))

    def log_likelihood(self, X, y, weights):
        '''
        ll = sum_{i=1}^Ny_iB^Tx_i - log(1 + e^{B^Tx_i})
        '''
        scores = np.dot(X, weights)
        ll = np.sum(y*scores - np.log(1 + np.exp(weights)))
        return ll