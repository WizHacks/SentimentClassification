import numpy as np

class NaiveBayes():
    '''Naive Bayes classifier with initial class names, labels of training data, vocabulary
    and whether the feature vector are integers or not'''
    def __init__(self, classes, labels, vocabulary):
        self.classes = classes
        self.vocabulary = vocabulary        
        self.prior = self.compute_prior(classes, labels)        
        print "naive bayes initialize: training set of size", (labels.shape[0])

    def compute_prior(self, classes, labels):
        '''Compute the prior for the given class names and target labels'''
        prior = np.zeros(len(classes))
        print "classes", classes
        for class_index in range(len(classes)):
            prior[class_index] = float(sum(label == class_index for label in labels))/labels.shape[0]
        print "priors:", prior
        return prior

    def compute_evidence(self, X):
        '''Compute the evidence for every word in the vocabulary'''
        print "computing evidence for data set of size:", X.shape[0]
        self.evidence = np.zeros(X.shape[0])   
        word_counts = np.sum(X, axis=0) # sums over the columns
        total_word_counts = np.sum(X) # sums over rows and columns
        for word_index in range(X.shape[0]):
            self.evidence[word_index] = float(word_counts[word_index])/total_word_counts 
        #print self.evidence    

    def compute_likelihood(self, X, y):
        '''Compute the log likelihood of each vocabulary word appearing in a data document
        with a corresponding label'''
        self.likelihood = np.zeros((len(self.classes), X.shape[1]))
        print "shape of likehood", self.likelihood.shape
        
        V = self.vocabulary.shape[0]
        print "Vocabulary size:", V
        for class_index in range(self.likelihood.shape[0]):
            w_c = sum(np.sum(X[fvi]) for fvi in range(X.shape[0]) if y[fvi] == class_index)
            print "Total words in class:", w_c
            for w in range(X.shape[1]):                
                wi_c = sum(X[fvi][w] for fvi in range(X.shape[0]) if y[fvi] == class_index)  
                #print "Total word i in class:", wi_c              
                self.likelihood[class_index][w] = float(wi_c + 1)/(w_c + V)         
        # print self.likelihood

    def train(self, X, y):
        '''Train the naive bayes classifier on training data with correctly labeled classes'''
        print "Training naive bayes classifer on %d documents ..." %(X.shape[0])
        #self.compute_evidence(X) not needed since constant factor        
        self.compute_likelihood(X,y)
        
    def discriminate(self, X):
        '''Determine which class labels are the most probable given likelihood and prior values'''
        class_posterior = np.zeros((len(self.classes), X.shape[0]))
        for class_index in range(class_posterior.shape[0]):
            class_likelihood = self.likelihood[class_index]

            for fvi in range(X.shape[0]):                                
                ml = 1
                for w in range(X[fvi].shape[0]):
                    if X[fvi][w] > 0:  
                        ml = ml + np.log(class_likelihood[w])
                class_posterior[class_index][fvi] = ml + np.log(self.prior[class_index])
        class_discriminate = np.zeros(X.shape[0])

        for fvi in range(X.shape[0]):
            class_max_prob = np.NINF
            for class_index in range(class_posterior.shape[0]):
                if class_posterior[class_index][fvi] > class_max_prob:
                    class_max_prob = class_posterior[class_index][fvi]
                    class_discriminate[fvi] = class_index
        #print class_discriminate
        return class_discriminate            

    def score(self, X, y):
        '''Score the naive bayes classifier on test data'''
        print "Testing naive bayes classifer on %d documents ..." %(X.shape[0])
        argmax_classes = self.discriminate(X)
        tp = fp = tn = fn = 0
        for fvi in range(X.shape[0]):               
            d = y[fvi]
            y_hat = argmax_classes[fvi]
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
        predictions = self.discriminate(X)
        print predictions
        return np.array(predictions.astype(int))
        