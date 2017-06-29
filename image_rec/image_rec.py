'''
 imageKNN.py (author: Shameer Sathar /  https://github.com/ssat335)
 
 Image identicality recommender system using an autoencoder-clustering model.
 
 Autoencoder Method:
  1) Train an autoencoder (simple/Conv) on training icc_images in 'db/icc_images_training' 
  2) Saves trained autoencoder, encoder, and decoder to 'db/models'

 Clustering Method:
  3) Using our trained encoder in 'db/models', we encode inventory icc_images in 'db/icc_images_inventory'
  4) Train kNN model using encoded inventory icc_images
  5) Encode query icc_images in 'query', and predict their NN using our trained kNN model
  6) Compute a score for each inventory encoding relative to our query encoding (centroid/closest)
  7) Make k-recommendations by cloning top-k inventory icc_images into 'answer'
'''
import sys, os, shutil
print("Python {0} on {1}".format(sys.version, sys.platform))
import numpy as np

from algo.utilities.image_utilities import ImageUtils
from algo.utilities.sorting import find_topk_unique
from algo.clustering.KNN import KNearestNeighbours
#from algo.autoencoders import simpleAE
#from algo.autoencoders import ConvAE

from keras.models import load_model

def main():
    project_root = os.path.dirname(__file__)
    sys.path.append(project_root)
    print("Project root: {0}".format(project_root))
    # ========================================
    # Set run settings
    # ========================================
    if 1:
        model_name = 'simpleAE'  # model folder
        flatten_before_encode = True
        flatten_after_encode = False
    elif 0:
        model_name = 'ConvAE'  # model folder
        flatten_before_encode = False
        flatten_after_encode = True
    else:
        raise Exception("Invalid model name which is not simpleAE nor ConvAE")

    model_extension_tag = '_encoder.h5'  # encoder model h5 tag
    img_shape = (100, 100)  # force resize of raw icc_images to (ypixels, xpixels)

    n_neighbors = 5  # number of nearest neighbours
    metric = "cosine"  # kNN metric (cosine only compatible with brute force)
    algorithm = "brute"  # search algorithm

    # Recommender mode:
    # 1 = nearest to centroid
    # 2 = nearest to any transaction point
    rec_mode = 2


    # ========================================
    # Generate expected file/folder paths and settings
    # ========================================
    # Assume project root directory to be directory of file
    project_root = os.path.dirname(__file__)

    # Query and answer folder
    query_dir = os.path.join(project_root, 'query')
    answer_dir = os.path.join(project_root, 'answer')

    # In image_repository folder
    db_dir = os.path.join(project_root, 'db')
    img_train_raw_dir = os.path.join(db_dir, 'img_train_raw')
    img_inventory_raw_dir = os.path.join(db_dir, 'img_inventory_raw')
    img_train_dir = os.path.join(db_dir, 'img_train')
    img_inventory_dir = os.path.join(db_dir, 'img_inventory')
    bin_train_dir = os.path.join(db_dir, 'bin_train')
    bin_inventory_dir = os.path.join(db_dir, 'bin_inventory')
    models_dir = os.path.join(db_dir, 'models')

    # In algorithms folder
    algo_dir = os.path.join(project_root, 'algo')
    autoencoders_dir = os.path.join(algo_dir, 'autoencoders')
    clustering_dir = os.path.join(algo_dir, 'clustering')
    utilities_dir = os.path.join(algo_dir, 'utilities')

    # Model encoder filename
    encoder_filename = os.path.join(models_dir, model_name + model_extension_tag)

    # Set info file
    info = {
        # Run settings
        "img_shape": img_shape,
        "flatten_before_encode": flatten_before_encode,

        # Directories
        "query_dir": query_dir,
        "answer_dir": answer_dir,

        "img_train_raw_dir": img_train_raw_dir,
        "img_inventory_raw_dir": img_inventory_raw_dir,
        "img_train_dir": img_train_dir,
        "img_inventory_dir": img_inventory_dir,
    }

    # Initialize image utilities (and register encoder)
    IU = ImageUtils()
    IU.configure(info)

    # ========================================
    #
    # Pre-process save/load training and inventory icc_images
    #
    # ========================================

    # Process and save
    process_save_icc_images = False
    if process_save_icc_images:
        # Training icc_images
        IU.raw2resized_load_save(raw_dir=img_train_raw_dir,
                                 processed_dir=img_train_dir,
                                 img_shape=img_shape)
        # Inventory icc_images
        IU.raw2resized_load_save(raw_dir=img_inventory_raw_dir,
                                 processed_dir=img_inventory_dir,
                                 img_shape=img_shape)


    # ========================================
    #
    # Train autoencoder
    #
    # ========================================
    train_autoencoder = False
    if train_autoencoder:
        print("Training the autoencoder...")

        # Load training icc_images to memory (resizes when necessary)
        x_data_train, train_filenames = \
            IU.raw2resizednorm_load(raw_dir=img_train_dir,
                                    img_shape=img_shape)
        print("x_data_train.shape = {0}".format(x_data_train.shape))


    # ========================================
    #
    # Perform clustering recommendation
    #
    # ========================================

    # Load inventory icc_images to memory (resizes when necessary)
    x_data_inventory, inventory_filenames = \
        IU.raw2resizednorm_load(raw_dir=img_inventory_dir,
                                img_shape=img_shape)
    print("x_data_inventory.shape = {0}".format(x_data_inventory.shape))

    # Load encoder
    encoder = load_model(encoder_filename)
    encoder.compile(optimizer='adam', loss='binary_crossentropy')  # set loss and optimizer

    # Encode our data, then flatten to encoding dimensions
    # We switch names for simplicity: inventory -> train, query -> test
    print("Encoding data and flatten its encoding dimensions...")
    if flatten_before_encode:  # Flatten the data before encoder prediction
        x_data_inventory = IU.flatten_img_data(x_data_inventory)
    x_train_kNN = encoder.predict(x_data_inventory)
    if flatten_after_encode:  # Flatten the data after encoder prediction
        x_train_kNN = IU.flatten_img_data(x_train_kNN)

    print("x_train_kNN.shape = {0}".format(x_train_kNN.shape))


    # =================================
    # Train kNN model
    # =================================
    print("Performing kNN to locate nearby items to user centroid points...")
    EMB = KNearestNeighbours()  # initialize embedding kNN class
    EMB.compile(n_neighbors = n_neighbors, algorithm = algorithm, metric = metric)  # compile kNN model
    EMB.fit(x_train_kNN)  # fit kNN


    # =================================
    # Perform kNN to the centroid query point
    # =================================
    while True:

        # Read items in query folder
        print("Reading query icc_images from query folder: {0}".format(query_dir))

        # Load query icc_images to memory (resizes when necessary)
        x_data_query, query_filenames = \
            IU.raw2resizednorm_load(raw_dir=query_dir,
                                    img_shape=img_shape)
        print("x_data_query.shape = {0}".format(x_data_query.shape))

        # Encode query icc_images
        if flatten_before_encode:  # Flatten the data before encoder prediction
            x_data_query = IU.flatten_img_data(x_data_query)
        x_test_kNN = encoder.predict(x_data_query)
        if flatten_after_encode:  # Flatten the data after encoder prediction
            x_test_kNN = IU.flatten_img_data(x_test_kNN)

        print("x_test_kNN.shape = {0}".format(x_test_kNN.shape))


        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute distances and indices for recommendation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if rec_mode == 1:  # kNN centroid transactions
            # Compute centroid point of the query encoding vectors (equal weights)
            x_test_kNN_centroid = np.mean(x_test_kNN, axis = 0)
            # Find nearest neighbours to centroid point
            distances, indices = EMB.predict(np.array([x_test_kNN_centroid]))

        elif rec_mode == 2:  # kNN all transactions
            # Find k nearest neighbours to all transactions, then flatten the distances and indices
            distances, indices = EMB.predict(x_test_kNN)
            distances = distances.flatten()
            indices = indices.flatten()
            # Pick k unique training indices which have the shortest distances any transaction point
            indices, distances = find_topk_unique(indices, distances, n_neighbors)

        else:
            raise Exception("Invalid method for making recommendations")

        print("distances.shape = {0}".format(distances.shape))
        print("indices.shape = {0}".format(indices.shape))

        # Make k-recommendations and clone most identical inventory icc_images to answer dir
        print("Cloning k-recommended inventory icc_images to answer folder '{0}'...".format(answer_dir))
        for i, (index, distance) in enumerate(zip(indices, distances)):
            print("({0}): index = {1}".format(i, index))
            print("({0}): distance = {1}".format(i, distance))

            for k_rec, ind in enumerate(index):

                # Extract inventory filename
                inventory_filename = inventory_filenames[ind]

                # Extract answer filename
                name, tag = IU.extract_name_tag(inventory_filename)
                answer_filename = os.path.join(answer_dir, name + '.' + tag)

                # Clone answer file to answer folder
                print("Cloning '{0}' to answer directory...".format(inventory_filename))
                shutil.copy(inventory_filename, answer_filename)

        # Wait for input (used for predicting other queries)
        c = input('Continue? Type `q` to break\n')
        if c == 'q':
            break


# Driver
if __name__ == "__main__":
    main()
