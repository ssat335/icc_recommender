"""

 tSNE.py  (author: Shameer Sathar / git: https://github.com/ssat335)

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold, datasets
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def plot_tsne(icc_images, X, filename):

    def imscatter(x, y, icc_images, ax=None, zoom=1.0):
        if ax is None:
            ax = plt.gca()
        x, y = np.atleast_1d(x, y)
        artists = []
        for x0, y0, img0 in zip(x, y, icc_images):
            im = OffsetImage(img0, zoom=zoom)
            ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()
        return artists

    def plot_embedding(X, imgs, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        plt.figure()
        ax = plt.subplot(111)
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], ".", fontdict={'weight': 'bold', 'size': 9})
        if hasattr(offsetbox, 'AnnotationBbox'):
            imscatter(X[:,0], X[:,1], imgs, zoom=0.1, ax=ax)

        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)

    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)
    plot_embedding(X_tsne, icc_images, "t-SNE embedding of icc_images")
    plt.savefig(filename, bbox_inches='tight')

# Driver
if __name__ == "__main__":
    digits = datasets.load_digits(n_class=6)
    icc_images = digits.icc_images
    X = digits.data
    run_tsne(icc_images, X, "tSNE_2.pdf")