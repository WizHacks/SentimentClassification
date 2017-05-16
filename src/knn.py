import numpy as np
import distance_metrics
import scipy.spatial.distance as scipy_dist

class KNN():
    '''KNN classifier with distance function, k value and p value if minkowski'''
    def __init__(self, distance_function, k=1, p=1):
        self.k = k
        self.distance_function = distance_function
        self.p = p
        print "knn initialize: k of %d" %(self.k)

    def train(self, X, y):
        '''Train the knn classifier on training data with correctly labeled classes'''
        print "Training knn classifer on %d documents ..." %(X.shape[0])
        # vector of means of dimension m where m is the number of features
        # this form of feature normalization is for vectors with different units
        # axis = 0, sums up the columns        
        # self.xmean = np.mean(X, axis=0)
        # self.xvar = np.var(X, axis=0)
        # self.xstd = np.sqrt(self.xvar)
        # print "Normalizing feature vectors"
        # [self.normalize(x) for x in X]
        # print "Normalized feature vectors"
        # Save training data and associated labels as part of model
        self.X = X
        self.y = y

    def score(self, X, y):
        '''Score the knn classifier on test data'''
        print "Testing knn classifer on %d documents ..." %(X.shape[0])
        tp = fp = tn = fn = 0
        for fvi in range(X.shape[0]):               
            d = y[fvi]
            # need to normalize X[fvi] as well if we normalized training data
            # self.normalize(X[fvi])
            y_hat = self.majority_voting(self.get_neighbors(X[fvi]))
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
        predictions = [self.majority_voting(self.get_neighbors(X[fvi])) for fvi in range(X.shape[0])]
        return np.array(predictions)
        
    # def normalize(self, x):
    #     # std = 0 when all values are identical
    #     # magnitude = euclidean norm
    #     magnitude = np.linalg.norm(x)
    #     for fi in range(len(x)):
    #         # divide vector by magnitude to make it unit length
    #         x[fi] = np.divide(x[fi], magnitude)
    #         # if self.xstd[fi] != 0:               
    #         #     x[fi] = (x[fi] - self.xmean[fi])/self.xstd[fi]

    def get_neighbors(self, x):
        '''Get the k closest neighbors of x using distance metric'''
        distances = []
        for xtrain in range(self.X.shape[0]):
            distance = self.distance_function(self.X[xtrain], x, self.p)
            distances.append((self.y[xtrain], distance))
        distances = np.array(distances)        
        # sort distances by 1st column
        distances = distances[distances[:,1].argsort()]
        # extract the first k neighbors' labels as nparray
        neigbors = distances[0:self.k, [0]]
        return neigbors

    def majority_voting(self, neighbors):
        '''Voting for the best classification given the classification of the neighbors'''
        class_votes = {}
        for neighbor in neighbors:
            if neighbor[0] in class_votes:
                class_votes[neighbor[0]] += 1
            else:
                class_votes[neighbor[0]] = 1        
        sorted_votes = sorted(class_votes.items(), reverse=True, key=lambda votes:votes[1])        
        return int(sorted_votes[0][0])