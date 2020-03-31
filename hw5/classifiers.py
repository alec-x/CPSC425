 #Starter code prepared by Borna Ghotbi for computer vision
 #based on MATLAB code by James Hay
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
'''This function will predict the category for every test image by finding
the training image with most similar features. Instead of 1 nearest
neighbor, you can vote based on k nearest neighbors which will increase
performance (although you need to pick a reasonable value for k). '''

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, neighbors):

    '''
    Parameters
        ----------
        train_image_feats:  is an N x d matrix, where d is the dimensionality of the feature representation.
        train_labels: is an N x l cell array, where each entry is a string 
        			  indicating the ground truth one-hot vector for each training image.
    	test_image_feats: is an M x d matrix, where d is the dimensionality of the
    					  feature representation. You can assume M = N unless you've modified the starter code.
        test_labels: is an M x 1 matrix.
        neighbors: is an int
    Returns
        -------
    	is an M x l cell array, where each row is a one-hot vector 
        indicating the predicted category for each test image.

    Usefull funtion:
    	
    	# You can use knn from sci-kit learn.
        # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    '''
    # Create KNN with n_neighbors being the number of neighbors referenced when predicting
    neigh = KNeighborsClassifier(n_neighbors = neighbors)
    # Train KNN classifier using training features and labels
    neigh.fit(train_image_feats, train_labels)

    # Predict testing labels using trained KNN model and testing features
    predicted_labels = neigh.predict(test_image_feats)
    return predicted_labels



'''This function will train a linear SVM for every category (i.e. one vs all)
and then use the learned linear classifiers to predict the category of
very test image. Every test feature will be evaluated with all 15 SVMs
and the most confident SVM will "win". Confidence, or distance from the
margin, is W*X + B where '*' is the inner product or dot product and W and
B are the learned hyperplane parameters. '''

def svm_classify(train_image_feats, train_labels, test_image_feats,C_val):

    '''
    Parameters
        ----------
        train_image_feats:  is an N x d matrix, where d is the dimensionality of the feature representation.
        train_labels: is an N x l cell array, where each entry is a string 
        			  indicating the ground truth one-hot vector for each training image.
    	test_image_feats: is an M x d matrix, where d is the dimensionality of the
    					  feature representation. You can assume M = N unless you've modified the starter code.

    Returns
        -------
    	is an M x l cell array, where each row is a one-hot vector 
        indicating the predicted category for each test image.

    Usefull funtion:
    	
    	# You can use svm from sci-kit learn.
        # Reference: https://scikit-learn.org/stable/modules/svm.html

    '''
    # Create svm model with regularization parameter C
    svm_model = svm.LinearSVC(C=C_val)
    # Train svm model using training features and labels
    svm_model.fit(train_image_feats, train_labels)

    # Predict the testing label given the testing features
    predicted_labels = svm_model.predict(test_image_feats)
    return predicted_labels

