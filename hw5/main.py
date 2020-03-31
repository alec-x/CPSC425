#Starter code prepared by Borna Ghotbi, Polina Zablotskaia, and Ariel Shann for Computer Vision
#based on a MATLAB code by James Hays and Sam Birch 

import numpy as np
from util import load, build_vocabulary, get_bags_of_sifts
from classifiers import nearest_neighbor_classify, svm_classify
import matplotlib.pyplot as plt
from os.path import dirname, basename

#For this assignment, you will need to report performance for sift features on two different classifiers:
# 1) Bag of sift features and nearest neighbor classifier
# 2) Bag of sift features and linear SVM classifier

#For simplicity you can define a "num_train_per_cat" vairable, limiting the number of
#examples per category. num_train_per_cat = 100 for intance.

#Sample images from the training/testing dataset. 
#You can limit number of samples by using the n_sample parameter.

print('Getting paths and labels for all train and test data\n')
train_image_paths, train_labels = load("sift/train")
test_image_paths, test_labels = load("sift/test")

''' Step 1: Represent each image with the appropriate feature
 Each function to construct features should return an N x d matrix, where
 N is the number of paths passed to the function and d is the 
 dimensionality of each image representation. See the starter code for
 each function for more details. '''
labels = []
# Extract label names
for i in range(15):
    # Extract first index where label is equal to i
    index = np.where(train_labels==i)[0][0]
    # Extract folder name one step up for the index with the label and append to labels
    labels.append(basename(dirname(train_image_paths[index])))

#TODO: You code build_vocabulary function in util.py
#TODO: You code get_bags_of_sifts function in util.py 

# Create function for doing getting features and saving
def bags_of_sifts_save(train_image_paths, test_image_paths, vocabSize, clusterSize, image_set_name):
    print "Generating K-means Clusters"
    kmeans = build_vocabulary(train_image_paths, vocabSize, clusterSize)
    print "Getting bag-of-SIFT Features"
    train_image_feats = get_bags_of_sifts(train_image_paths, kmeans)
    test_image_feats = get_bags_of_sifts(test_image_paths, kmeans)
    print "Creating np data files"
    train_feats_file = open("train_image_feats_" + str(image_set_name) + ".npy", "wb")
    test_feats_file = open("test_image_feats_" + str(image_set_name) + ".npy", "wb")
    train_labels_file = open("train_image_labels_" + str(image_set_name) + ".npy", "wb")
    test_labels_file = open("test_image_labels_" + str(image_set_name) + ".npy", "wb")
    np.save(train_feats_file, train_image_feats)
    np.save(test_feats_file, test_image_feats)
    np.save(train_labels_file, train_labels)
    np.save(test_labels_file, test_labels)
    train_feats_file.close()
    test_feats_file.close()
    train_labels_file.close()
    test_labels_file.close()
    return

# bags_of_sifts_save(train_image_paths, test_image_paths, 200, 100, "all")


print "Load saved features for training and testing"
train_image_feats = np.load("train_image_feats_all.npy")
test_image_feats = np.load("test_image_feats_all.npy")
train_labels = np.load("train_image_labels_all.npy")
test_labels = np.load("test_image_labels_all.npy")
'''
# average histograms

# For each picture histogram, increment the correct category histogram. 
categorized_feats = np.zeros((15,train_image_feats.shape[1]))
for i,im_feats in enumerate(train_image_feats):
    categorized_feats[train_labels[i]] += im_feats

print "Generating Graphs"
# Save the histogram for each category
for i in range(15):
    plt.bar(range(train_image_feats.shape[1]), categorized_feats[i])
    plt.title("Histogram of features for " + labels[i])
    plt.savefig('./q4histograms/' + labels[i] + '.png')
    plt.clf()
#If you want to avoid recomputing the features while debugging the
#classifiers, you can either 'save' and 'load' the extracted features
#to/from a file.
'''
''' Step 2: Classify each test image by training and using the appropriate classifier
 Each function to classify test features will return an N x l cell array,
 where N is the number of test cases and each entry is a string indicating
 the predicted one-hot vector for each test image. See the starter code for each function
 for more details. '''

print('Using nearest neighbor classifier to predict test set categories\n')
#TODO: YOU CODE nearest_neighbor_classify function from classifers.py
pred_labels_knn = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, 5)
knn_acc = np.mean(pred_labels_knn == test_labels)

print('Using support vector machine to predict test set categories\n')
#TODO: YOU CODE svm_classify function from classifers.py
pred_labels_svm = svm_classify(train_image_feats, train_labels, test_image_feats, 15)
svm_acc = np.mean(pred_labels_svm == test_labels)

print('---Evaluation---\n')

print "knn accuracy: " + str(knn_acc)
print "svm accuracy: " + str(svm_acc)
print np.sum(pred_labels_knn[pred_labels_knn==0] == test_labels[test_labels==0])

confusion_knn = np.zeros((15,15))
confusion_svm = np.zeros((15,15))

for i in range(len(test_labels)):
        confusion_knn[pred_labels_knn[i]][test_labels[i]] += 1
        confusion_svm[pred_labels_svm[i]][test_labels[i]] += 1

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion_knn)
fig.colorbar(cax)
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
plt.xticks(rotation=90)
plt.title("Confusion Matrix for KNN")
plt.tight_layout()
plt.savefig('./q6cm/knn.png')
plt.clf()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
cax2 = ax2.matshow(confusion_svm)
fig2.colorbar(cax2)
ax2.set_xticks(np.arange(len(labels)))
ax2.set_yticks(np.arange(len(labels)))
ax2.set_xticklabels(labels)
ax2.set_yticklabels(labels)
ax2.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
plt.xticks(rotation=90)
plt.title("Confusion Matrix for SVM")
plt.tight_layout()
plt.savefig('./q6cm/svm.png')
plt.clf()
# Step 3: Build a confusion matrix and score the recognition system for 
#         each of the classifiers.
# TODO: In this step you will be doing evaluation. 
# 1) Calculate the total accuracy of your model by counting number
#   of true positives and true negatives over all. 
# 2) Build a Confusion matrix and visualize it. 
#   You will need to convert the one-hot format labels back
#   to their category name format.


# Interpreting your performance with 100 training examples per category:
#  accuracy  =   0 -> Your code is broken (probably not the classifier's
#                     fault! A classifier would have to be amazing to
#                     perform this badly).
#  accuracy ~= .10 -> Your performance is chance. Something is broken or
#                     you ran the starter code unchanged.
#  accuracy ~= .40 -> Rough performance with bag of SIFT and nearest
#                     neighbor classifier. 
#  accuracy ~= .50 -> You've gotten things roughly correct with bag of
#                     SIFT and a linear SVM classifier.
#  accuracy >= .60 -> You've added in spatial information somehow or you've
#                     added additional, complementary image features. This
#                     represents state of the art in Lazebnik et al 2006.
#  accuracy >= .85 -> You've done extremely well. This is the state of the
#                     art in the 2010 SUN database paper from fusing many 
#                     features. Don't trust this number unless you actually
#                     measure many random splits.
#  accuracy >= .90 -> You used modern deep features trained on much larger
#                     image databases.
#  accuracy >= .96 -> You can beat a human at this task. This isn't a
#                     realistic number. Some accuracy calculation is broken
#                     or your classifier is cheating and seeing the test
#                     labels.
