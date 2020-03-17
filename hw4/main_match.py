import hw_utils as utils
import matplotlib.pyplot as plt
import sys

def main():
    # 3 arguments, threshold, orient agreement, and scale agreement. Orientation in degrees
    assert(len(sys.argv) == 4)
    threshold = float(sys.argv[1])
    orientThreshold = float(sys.argv[2])
    scaleThreshold = float(sys.argv[3])

    # Q1.3
    '''
    # matching with no ransac

    plt.figure(figsize=(20, 20))
    im = utils.Match('./data/scene', './data/basmati', ratio_thres=threshold)
    plt.title('Match')
    plt.imshow(im)
    '''
    # Q1.4
    # matching with no ransac
    
    plt.figure(figsize=(20, 20))
    im = utils.Match('./data/library', './data/library2', ratio_thres=threshold)
    plt.title('Match')
    plt.imshow(im)
    # matching with ransac
    
    plt.figure(figsize=(20, 20))
    im = utils.MatchRANSAC(
        './data/library', './data/library2',
        ratio_thres=threshold, orient_agreement=orientThreshold, scale_agreement=scaleThreshold)
    plt.title('MatchRANSAC')
    plt.imshow(im)
    
if __name__ == '__main__':
    main()
