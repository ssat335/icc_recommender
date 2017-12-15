"""

 identical_icc_images_TL.py  (author: Shameer Sathar / git: https://github.com/ssat335)

 We find identical icc_images in a image_repository by using transfer learning
 using a pre-trained VGG image classifier. We plot the 5 most identical
 icc_images for each image in the image_repository, and plot the tSNE for all
 our image feature vectors.

"""
import sys, os
import numpy as np
from keras.preprocessing import image
from keras.models import Model
sys.path.append("src")

from vgg19 import VGG19
from imagenet_utils import preprocess_input
from plot_utils import plot_query_answer
from sort_utils import find_topk_unique
from kNN import kNN
from tSNE import plot_tsne

def main():
    # ================================================
    # Load pre-trained model and remove higher level layers
    # ================================================
    print("Loading VGG19 pre-trained model...")
    base_model = VGG19(weights='imagenet')
    model = Model(input=base_model.input,
                  output=base_model.get_layer('block4_pool').output)

    # ================================================
    # Read icc_images and convert them to feature vectors
    # ================================================
    imgs, filename_heads, X = [], [], []
    path = "db"
    print("Reading icc_images from '{}' directory...\n".format(path))
    for f in os.listdir(path):

        # Process filename
        filename = os.path.splitext(f)  # filename in directory
        filename_full = os.path.join(path,f)  # full path filename
        head, ext = filename[0], filename[1]
        if ext.lower() not in [".jpg", ".jpeg"]:
            continue

        # Read image file
        img = image.load_img(filename_full, target_size=(224, 224))  # load
        imgs.append(np.array(img))  # image
        filename_heads.append(head)  # filename head

        # Pre-process for model input
        img = image.img_to_array(img)  # convert to array
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = model.predict(img).flatten()  # features
        X.append(features)  # append feature extractor

    X = np.array(X)  # feature vectors
    imgs = np.array(imgs)  # icc_images
    print("imgs.shape = {}".format(imgs.shape))
    print("X_features.shape = {}\n".format(X.shape))

    # ===========================
    # Find k-nearest icc_images to each image
    # ===========================
    n_neighbours = 5 + 1  # +1 as itself is most identical
    knn = kNN()  # kNN model
    knn.compile(n_neighbors=n_neighbours, algorithm="brute", metric="cosine")
    knn.fit(X)

    # ==================================================
    # Plot recommendations for each image in image_repository
    # ==================================================
    output_rec_dir = os.path.join("output", "rec")
    if not os.path.exists(output_rec_dir):
        os.makedirs(output_rec_dir)
    n_imgs = len(imgs)
    ypixels, xpixels = imgs[0].shape[0], imgs[0].shape[1]
    for ind_query in range(n_imgs):

        # Find top-k closest image feature vectors to each vector
        print("[{}/{}] Plotting identical image recommendations for: {}".format(ind_query+1, n_imgs, filename_heads[ind_query]))
        distances, indices = knn.predict(np.array([X[ind_query]]))
        distances = distances.flatten()
        indices = indices.flatten()
        indices, distances = find_topk_unique(indices, distances, n_neighbours)

        # Plot recommendations
        rec_filename = os.path.join(output_rec_dir, "{}_rec.png".format(filename_heads[ind_query]))
        x_query_plot = imgs[ind_query].reshape((-1, ypixels, xpixels, 3))
        x_answer_plot = imgs[indices].reshape((-1, ypixels, xpixels, 3))
        plot_query_answer(x_query=x_query_plot,
                          x_answer=x_answer_plot[1:],  # remove itself
                          filename=rec_filename)

    # ===========================
    # Plot tSNE
    # ===========================
    output_tsne_dir = os.path.join("output")
    if not os.path.exists(output_tsne_dir):
        os.makedirs(output_tsne_dir)
    tsne_filename = os.path.join(output_tsne_dir, "tsne.png")
    print("Plotting tSNE to {}...".format(tsne_filename))
    plot_tsne(imgs, X, tsne_filename)

# Driver
if __name__ == "__main__":
    main()