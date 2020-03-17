import hw_utils as utils
import matplotlib.pyplot as plt
import sys

def main():
    # 3 arguments, threshold, orient agreement, and scale agreement
    assert(len(sys.argv) == 4)
    threshold = float(sys.argv[1])
    orientThreshold = float(sys.argv[2])
    scaleThreshold = float(sys.argv[3])
    # Test run matching with no ransac
    plt.figure(figsize=(20, 20))

    im = utils.Match('./data/scene', './data/book', ratio_thres=threshold)
    plt.title('Match')
    plt.imshow(im)

    # Test run matching with ransac
    plt.figure(figsize=(20, 20))
    im = utils.MatchRANSAC(
        './data/scene', './data/basmati',
        ratio_thres=threshold, orient_agreement=orientThreshold, scale_agreement=scaleThreshold)
    plt.title('MatchRANSAC')
    plt.imshow(im)

if __name__ == '__main__':
    main()
