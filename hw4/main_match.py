import hw_utils as utils
import matplotlib.pyplot as plt
import sys

def main():
    assert(len(sys.argv) == 2)
    # Test run matching with no ransac
    plt.figure(figsize=(20, 20))

    im = utils.Match('./data/scene', './data/book', ratio_thres=float(sys.argv[1]))
    plt.title('Match')
    plt.imshow(im)

    # Test run matching with ransac
    plt.figure(figsize=(20, 20))
    im = utils.MatchRANSAC(
        './data/scene', './data/basmati',
        ratio_thres=0.6, orient_agreement=30, scale_agreement=0.5)
    plt.title('MatchRANSAC')
    plt.imshow(im)

if __name__ == '__main__':
    main()
