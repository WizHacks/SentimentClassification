from string import maketrans, punctuation
import numpy as np

def strip_punct(documents):
    '''Strip the punctuation from the documents'''
    print "Stripping the punctuation from the documents ..."        
    data = []
    table = maketrans("","")
    for doc in documents:  
        mod_doc = doc.translate(table, punctuation)
        data.append(mod_doc) 
    print "Returning %d documents stripped of punctuation..." %(len(data))
    return np.array(data)

def remove_stopwords(documents):
    '''Remove the stopwords from the documents'''
    print "Removing the stop words from the documents ..."        
    with open('stopwords.txt') as f:
        stopwords = f.readlines()
    stopwords = [word.strip() for word in stopwords]
    stopwords = set(stopwords)
    data = []    
    for doc in documents:  
        mod_doc_words = doc.split()
        mod_doc_words_result = [word for word in mod_doc_words if word not in stopwords] 
        mod_doc = " ".join(mod_doc_words_result)
        data.append(mod_doc)
    print "Returning %d documents with stop words removed..." %(len(data))
    return np.array(data)

def get_vocabulary(documents):
    '''Get the vocabulary for the documents'''
    print "Creating vocabulary for documents ..."  
    vocabulary = {}   
    for doc in documents:   
        # delimits on consecutive whitespace, removes crnl    
        tokens = doc.split()
        for token in tokens:            
            if token not in vocabulary:
                vocabulary[token] = 0
    print "Returning vocabulary of size %d..." %(len(vocabulary))  
    return np.array(vocabulary.keys())

def normalize(x):
    '''Normalize the data to unit length'''
    # magnitude = euclidean norm
    magnitude = np.linalg.norm(x)
    for fi in range(len(x)):
        # divide vector by magnitude to make it unit length
        x[fi] = np.divide(x[fi], magnitude)

class BinaryModel():
    '''Transform data with given vocabulary into binary model of feature vectors'''
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary        
        print "BinaryModel with vocabulary of size: %d" %(self.vocabulary.shape[0])

    def create_binary_model(self, documents):
        '''Do the work for creating the binary model feature vectors for a set of data'''
        print "Creating binary model feature vectors for %d documents..." %(len(documents))   
        feature_vectors = []
        for doc in documents:
            vocabulary = {}
            tokens = doc.split()
            
            for token in tokens:
                if token not in vocabulary:
                    vocabulary[token] = 1
              
            feature_vector = []
            for token in self.vocabulary:
                if token not in vocabulary:
                    feature_vector.append(0)
                else:
                    feature_vector.append(1)         
            feature_vectors.append(feature_vector)

        print "Returning binary model feature vectors of size %d for %d documents..." %(len(feature_vectors[0]), len(feature_vectors))   
        #print np.array(feature_vectors)
        return np.array(feature_vectors)

class BagOfWordsModel():
    '''Transform data with given vocabulary into bow model of feature vectors'''
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary        
        print "BagOfWordsModel with vocabulary of size: %d" %(self.vocabulary.shape[0])

    def create_bow_model(self, documents):
        '''Do the work for creating the bow model feature vectors for a set of data'''
        print "Creating bow model feature vectors for %d documents..." %(len(documents))   
        feature_vectors = []
        for doc in documents:
            vocabulary = {}
            tokens = doc.split()
            
            for token in tokens:
                if token not in vocabulary:
                    vocabulary[token] = 1
                else:
                    vocabulary[token] += 1                
              
            feature_vector = []
            for token in self.vocabulary:
                if token not in vocabulary:
                    feature_vector.append(0)
                else:
                    feature_vector.append(vocabulary[token])                     
            feature_vectors.append(feature_vector)       
        print "Returning bow model feature vectors of size %d for %d documents..." %(len(feature_vectors[0]), len(feature_vectors))   
        return np.array(feature_vectors)

class TFIDFModel():
    '''Transform data with given vocabulary into tf-idf model of feature vectors.
    The number of documents with term t in it is saved so it can be accessed when creating 
    feature vectors for new test documents on the fly'''
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary  
        self.doc_terms_idf = None      
        print "TFIDFModel with vocabulary of size: %d" %(self.vocabulary.shape[0])

    def create_tfidf_model(self, documents):
        '''Do the work for creating the tf-idf model feature vectors for a set of data'''
        print "Creating tf-idf model feature vectors for %d documents..." %(len(documents))       
        feature_vectors = []
        if self.doc_terms_idf == None:
            self.doc_terms_idf = {}
        for doc in documents:
            vocabulary = {}
            tokens = doc.split()
            
            for token in tokens:
                if token not in vocabulary:
                    vocabulary[token] = 1
                else:
                    vocabulary[token] += 1

            total_tokens = len(tokens)  
            #print "total tokens in doc:", total_tokens
            feature_vector = []
            for token in self.vocabulary:                
                if token not in vocabulary:
                    feature_vector.append(0)
                else:
                    # add tf value to feature vector first
                    feature_vector.append(float(vocabulary[token])/total_tokens)                        
                    if token in self.doc_terms_idf:
                        self.doc_terms_idf[token] += 1
                    else:
                        self.doc_terms_idf[token] = 1                        
            feature_vectors.append(feature_vector) 
        #print self.doc_terms_idf.values()
        #print feature_vectors
        #feature_vectors = np.array(feature_vectors)    
        for doc_index in range(documents.shape[0]):
            w = -1 # used as index for word in feature vector
            for token in self.vocabulary: 
                w = w + 1                       
                #if doc_terms_idf[token]
                idf = np.log(1 + (float(documents.shape[0]))/(self.doc_terms_idf[token]))
                #print idf
                if feature_vectors[doc_index][w] > 0:
                    tf = (1 + np.log(feature_vectors[doc_index][w]))
                else:
                    tf = 0
                feature_vectors[doc_index][w] = tf*idf # wiki def 2                           
        print "Returning tf-idf model feature vectors of size %d for %d documents..." %(len(feature_vectors[0]), len(feature_vectors))   
        #print np.array(feature_vectors)
        return np.array(feature_vectors)
