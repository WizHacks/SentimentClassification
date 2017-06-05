import numpy as np

class LogisticRegression():
    '''LogisticRegression classifier'''
    def __init__(self, learning_rate, use_intercept=False):
        self.learning_rate = learning_rate
        self.use_intercept = use_intercept
        print "Logistic Regression initialize with learning rate %f, and intercept %s" %(learning_rate, use_intercept)

    def train(self, X, y):
        '''Train the logistic regression classifier on training data with correctly labeled classes'''
        print "Training logistic regression classifer on %d documents ..." %(X.shape[0])        
        '''
        >>> a = np.array([[1],[2],[3]])
        >>> b = np.array([[2],[3],[4]])
        >>> np.hstack((a,b))
        array([[1, 2],
               [2, 3],
               [3, 4]])
        '''
        # Add ones to the end of all the feature vectors
        if self.use_intercept:
            intercept = np.ones(((X.shape[0]), 1))
            X = np.hstack((intercept, X))

        # length of feature vector weights
        self.weights = np.zeros(X.shape[1])    

        for fvi in range(X.shape[0]):
            scores = np.dot(X[fvi], self.weights)
            # starting predictions
            # 1 + e^{B^Tx_i}
            predictions = self.sigmoid(scores)            

            # Update weights using gradient ascent, unrounded for training purposes
            output_error_signal = y[fvi] - predictions
            '''
            \nabla ll = X^T(Y-predictions)
            '''
            gradient = np.dot(X[fvi].T, output_error_signal)            
            self.weights += (self.learning_rate * gradient)

            # # Print log-likelihood every so often--global max
            # if fvi % 100 == 0:
            #     print log_likelihood(X[fvi], y[fvi], self.weights)

    def score(self, X, y):
        '''Score the logistic regression classifier on test data'''
        print "Testing logistic regression classifer on %d documents ..." %(X.shape[0])
        tp = fp = tn = fn = 0
        predictions = self.predict(X)
        for prediction in range(predictions.shape[0]):               
            d = y[prediction]
            y_hat = predictions[prediction]
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
        if self.use_intercept:
            X = np.hstack((np.ones((X.shape[0],1)), X))
        scores = np.dot(X, self.weights)
        # rounded for prediction purposes
        predictions = np.round(self.sigmoid(scores))
        return predictions
        
    def sigmoid(self, X):
        '''
        Might throw an overflow exception, but handled by numpy
        autoset to 0
        '''
        return 1/(1 + np.exp(-X))

    def log_likelihood(self, X, y, weights):
        '''
        ll = sum_{i=1}^Ny_iB^Tx_i - log(1 + e^{B^Tx_i})
        '''
        scores = np.dot(X, weights)
        ll = np.sum(y*scores - np.log(1 + np.exp(weights)))
        return ll
