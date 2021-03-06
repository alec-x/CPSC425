import numpy as np
import cv2
import math
import random

def RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement):
    """
    This function takes in `matched_pairs`, a list of matches in indices
    and return a subset of the pairs using RANSAC.
    Inputs:
        matched_pairs: a list of tuples [(i, j)],
            indicating keypoints1[i] is matched
            with keypoints2[j]
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        *_agreement: thresholds for defining inliers, floats
    Output:
        largest_set: the largest consensus set in [(i, j)] format

    HINTS: the "*_agreement" definitions are well-explained
           in the assignment instructions.
    """
    assert isinstance(matched_pairs, list)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    ## START
    subsets = [] # list of lists of match pairs (list) so list of lists of lists
    oThresh = orient_agreement/180.0*math.pi # Convert degrees to radians

    for _ in range(10):
        randomMatch = random.choice(matched_pairs)

        baseAngle = (keypoints1[randomMatch[0]][3] - keypoints2[randomMatch[1]][3]) % (2.0*math.pi)
        baseScale = keypoints2[randomMatch[1]][2]/keypoints1[randomMatch[0]][2]
        
        goodKeyPoints = []
        for match in matched_pairs:
            matchAngle = (keypoints1[match[0]][3] - keypoints2[match[1]][3]) % (2.0*math.pi)
            matchScale = keypoints2[match[1]][2]/keypoints1[match[0]][2]
            '''
            print "..."
            print str(baseAngle - oThresh) + " " + str(matchAngle) + " " + str(baseAngle + oThresh)
            '''
            if baseAngle - oThresh <= matchAngle <= baseAngle + oThresh:
                if baseScale - scale_agreement <= matchScale <= baseScale + scale_agreement:
                    goodKeyPoints.append(match)
        
        subsets.append(goodKeyPoints)
    largest_set = max(subsets, key=len)

    ## EN
    assert isinstance(largest_set, list)
    return largest_set

def FindBestMatches(descriptors1, descriptors2, threshold):
    """
    This function takes in descriptors of image 1 and image 2,
    and find matches between them. See assignment instructions for details.
    Inputs:
        descriptors: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
    Outputs:
        matched_pairs: a list in the form [(i, j)] where i and j means
                       descriptors1[i] is matched with descriptors2[j].
    """
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    ## START
    ## the following is just a placeholder to show you the output format
    matched_pairs = []
    indexOf1 = 0
    for i in descriptors1:
        currAngles = [] # holds all comparisons to current descriptor
        
        # calculate all comparisons to current descriptor
        for j in descriptors2:
            currAngles.append(math.acos(np.dot(i, j)))
        
        # Sort angles from smallest to largest
        sortedAngles = sorted(currAngles)
        
        if sortedAngles[0]/sortedAngles[1] < threshold:
            # Find index of i in descriptors1, and index of smallest sorted angle in current angles
            matched_pairs.append([indexOf1, currAngles.index(sortedAngles[0])])
        indexOf1 += 1
    # num = 5
    # matched_pairs = [[i, i] for i in range(num)]
    ## END
    return matched_pairs


def KeypointProjection(xy_points, h):
    """
    This function projects a list of points in the source image to the
    reference image using a homography matrix `h`.
    Inputs:
        xy_points: numpy array, (num_points, 2)
        h: numpy array, (3, 3), the homography matrix
    Output:
        xy_points_out: numpy array, (num_points, 2), input points in
        the reference frame.
    """
    assert isinstance(xy_points, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert xy_points.shape[1] == 2
    assert h.shape == (3, 3)

    # START   

    # Initialize out array to have same shape as in array
    xy_points_out = np.empty_like(xy_points) 

    # Append 1 column to xy_points
    xy_points = np.hstack((xy_points,np.ones((xy_points.shape[0],1))))

    # Treating each point separately
    for i in range(xy_points.shape[0]):
        # Convert point to homogenous coordinate by multiplying with h matrix
        projCoord = np.matmul(h,xy_points[i,:])
        
        # Divide x,y coord by z coord and assign to output array
        xy_points_out[i] = np.array([projCoord[0]/projCoord[2], projCoord[1]/projCoord[2]])
    # END
    return xy_points_out

def RANSACHomography(xy_src, xy_ref, num_iter, tol):
    """
    Given matches of keypoint xy coordinates, perform RANSAC to obtain
    the homography matrix. At each iteration, this function randomly
    choose 4 matches from xy_src and xy_ref.  Compute the homography matrix
    using the 4 matches.  Project all source "xy_src" keypoints to the
    reference image.  Check how many projected keypoints are within a `tol`
    radius to the coresponding xy_ref points (a.k.a. inliers).  During the
    iterations, you should keep track of the iteration that yields the largest
    inlier set. After the iterations, you should use the biggest inlier set to
    compute the final homography matrix.
    Inputs:
        xy_src: a numpy array of xy coordinates, (num_matches, 2)
        xy_ref: a numpy array of xy coordinates, (num_matches, 2)
        num_iter: number of RANSAC iterations.
        tol: float
    Outputs:
        h: The final homography matrix.
    """
    assert isinstance(xy_src, np.ndarray)
    assert isinstance(xy_ref, np.ndarray)
    assert xy_src.shape == xy_ref.shape
    assert xy_src.shape[1] == 2
    assert isinstance(num_iter, int)
    assert isinstance(tol, (int, float))
    tol = tol*1.0

    # START
    maxInliers = 0
    h = None
    
    # Tunable param in case we wanted to use more than 4 matches
    numSample = 4
    
    # number samples in xy_src used to randomly select indices
    num_matches = xy_src.shape[0]

    for _ in range(num_iter):
        # Randomly select indices numSample number of times
        randIndexes = [random.randint(0,num_matches-1) for _ in range(numSample)]

        # Create empty arrays for the randomly selected indices and build them
        # by stacking the retrieved array slices
        xy_src_i = np.array([]).reshape(0,2)
        xy_ref_i = np.array([]).reshape(0,2)
        for index in randIndexes:
            xy_src_i = np.vstack((xy_src_i, xy_src[index]))
            xy_ref_i = np.vstack((xy_ref_i, xy_ref[index]))

        # Generate homography matrix with current samples
        currh, _ = cv2.findHomography(xy_src_i, xy_ref_i)

        numInliers = 0
        for i in range(len(xy_src)):
            # Calculate the projected, then the output point using source and h matrix
            projPoint = np.matmul(currh,np.hstack((xy_src[i,:],1)))
            outPoint = np.array([projPoint[0]/projPoint[2], projPoint[1]/projPoint[2]])
            
            # Calculate distance
            outX = outPoint[0]
            outY = outPoint[1]
            refX = xy_ref[i,0]
            refY = xy_ref[i,1]
            dist = math.sqrt((outX - refX)**2 + (outY-refY)**2)

            # Evaluate distance against tolerance
            if dist <= tol:
                numInliers += 1
        
        # If current h is best, update maximum inliers achieved and output h
        if numInliers > maxInliers:
            h = currh
            maxInliers = numInliers
    # END
    assert isinstance(h, np.ndarray)
    assert h.shape == (3, 3)
    return h


def FindBestMatchesRANSAC(
        keypoints1, keypoints2,
        descriptors1, descriptors2, threshold,
        orient_agreement, scale_agreement):
    """
    Note: you do not need to change this function.
    However, we recommend you to study this function carefully
    to understand how each component interacts with each other.

    This function find the best matches between two images using RANSAC.
    Inputs:
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        descriptors1, 2: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
        orient_agreement: in degrees, say 30 degrees.
        scale_agreement: in floating points, say 0.5
    Outputs:
        matched_pairs_ransac: a list in the form [(i, j)] where i and j means
        descriptors1[i] is matched with descriptors2[j].
    Detailed instructions are on the assignment website
    """
    orient_agreement = float(orient_agreement)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    matched_pairs = FindBestMatches(
        descriptors1, descriptors2, threshold)
    matched_pairs_ransac = RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement)
    return matched_pairs_ransac
