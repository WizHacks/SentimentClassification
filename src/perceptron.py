import numpy as np

class Perceptron():
    '''Perceptron classifier with initial weights, bias, and learning rate'''
    def __init__(self, bias = 1.0, learning_rate = 1.0):
        self.bias = bias     
        self.learning_rate = learning_rate
        print "perceptron initialize: bias of %.2f, learning_rate of %f" %(self.bias, self.learning_rate)

    def train(self, X, y):
        '''Train the perceptron classifier on training data with correctly labeled classes'''
        print "Training perceptron classifer on %d documents ..." %(X.shape[0])
        # length of feature vector weights   
        self.weights = np.zeros(X.shape[1]) 
        
        for fvi in range(y.size):               
            d = y[fvi]
            y_hat = self.sgn(np.dot(self.weights, X[fvi]) + self.bias)        
            self.update_weights(d, y_hat, X[fvi])
            self.update_bias(d, y_hat, X[fvi])
        print self.weights

    def score(self, X, y):
        '''Score the perceptron classifier on test data'''
        print "Testing perceptron classifer on %d documents ..." %(X.shape[0])
        tp = fp = tn = fn = 0
        predictions = self.predict(X)
        for prediction in range(predictions.shape[0]):               
            d = y[prediction]
            y_hat = predictions[prediction]
        # for fvi in range(X.shape[0]):               
        #     d = y[fvi]
        #     y_hat = self.sgn(np.dot(self.weights, X[fvi]) + self.bias)
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

    def sgn(self, v):
        '''sign function'''
        # print v
        # v[v >= 0] = 1
        # v[v < 0 ] = 1
        # return v this is a single value but in this case we want array
        # 
        # for single value sgn(x)
        if v >= 0:
            return 1
        return 0

    def update_weights(self, d, y_hat, x):
        '''Update the weights given desired output and our output'''
        self.weights = self.weights + np.dot(self.learning_rate*(d-y_hat), x)    

    def update_bias(self, d, y_hat, x):
        '''Since bias is not part of the feature vector and weights, it needs
        its own update'''
        self.bias = self.bias + self.learning_rate*(d-y_hat)           

    def predict(self, X):
        # '''Given a new set of test documents, predict their labels'''
        print "Predicting on %d documents ..." %(X.shape[0])
        predictions = [self.sgn(np.dot(self.weights, X[fvi]) + self.bias) for fvi in range(X.shape[0])]
        return np.array(predictions)
        # same? no sgn works on one dot product at a time
        # print X.shape
        # print self.weights.shape
        # predictions = self.sgn(np.dot(X, self.weights) + self.bias)
        # return np.array(predictions)
        