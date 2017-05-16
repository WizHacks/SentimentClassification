import numpy as np

class Rocchio():
    '''Rocchio classifier with target classes, distance function and p value if minkowski '''
    def __init__(self, classes, distance_function, p=1):
        self.classes = classes
        self.distance_function = distance_function
        self.p = p
        print "rocchio initialize: number of classes %d" %(len(self.classes))

    def train(self, X, y):
        '''Train the rocchio classifier on training data with correctly labeled classes'''
        print "Training rocchio classifer on %d documents ..." %(X.shape[0])
        # vector of means of dimension m where m is the number of features
        # axis = 0, sums up the columns                
        self.class_means = np.zeros((len(self.classes), X.shape[1]))
        class_count = {}
        for xi in range(X.shape[0]):            
            for i in range(X.shape[1]):
                self.class_means[y[xi]][i] += X[xi][i]       
            if y[xi] in class_count:
                class_count[y[xi]] +=1
            else:
                class_count[y[xi]] = 1
        for mean_num in range(len(self.classes)):         
            self.class_means[mean_num] = np.divide(self.class_means[mean_num],class_count[mean_num])

    def score(self, X, y):
        '''Score the rocchio classifier on test data'''
        print "Testing rocchio classifer on %d documents ..." %(X.shape[0])
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
        
    def get_nearest_centroid(self, x):
        '''Get the classification whose mean is closest to x'''
        distances = []
        for mean_num in range(self.class_means.shape[0]):
            distance = self.distance_function(self.class_means[mean_num], x, self.p)
            distances.append((mean_num, distance))
        distances = np.array(distances)
        # sort distances by 1st column
        distances = distances[distances[:,1].argsort()]
        # extract the closest centroid label as prediction
        return int(distances[0:1, [0]])