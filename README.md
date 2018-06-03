# Similar ICC images finder using transfer learning

A pretrained image classification VGG19 DNN network, is used to find indentical images for a given ICC image. The last layers of the network is removed and the dissected model is used to convert the images into feature vectors for similarity comparison to produce closes matching ICC images. Since we are using a pretrained network, no further training is required.

In addition to making ICC image recommendation, we can also visualize the image feature vectors by mapping the high-dimensional vectors onto a 2-dimensional manifold using the t-SNE algorithm to get a sense of how "far away" images are from each other in the feature space:

Usage:

1. Prepare the image database.

2. Take the VGG model and remove its last layers.

3. Convert our image database into feature vectors using our dissected VGG model. If the output layer of the dissected model are convolutional filters then flatten the filters and append them make a single vector.

4. Compute similarities between our image feature vectors using an inner-product such as cosine similarity or euclidean distance

5. For each image, select the top-k images that have the highest similarity scores to build the recommendation


### Usage:

1. Place your database of images into the `db` directory.

2. Run the command:

    > python similar_images_TL.py

    All output from running this code will be placed in the `output` directory. There will be a `tsne.png` plot for the t-SNE visualization of your database image embeddings, as well as a `rec` directory containing the top `k = 5` similar image recommendations for each image in your database.

If the program is running properly, you should see something of the form:

```
Loading VGG19 pre-trained model...
Reading images from 'db' directory...

imgs.shape = (39, 224, 224, 3)
X_features.shape = (39, 100352)

[1/39] Plotting similar image recommendations for: ....
[2/39] Plotting similar image recommendations for: ....
[3/39] Plotting similar image recommendations for: ...
...
...
...
[38/39] Plotting similar image recommendations for: ...
[39/39] Plotting similar image recommendations for: ...
Plotting tSNE to output/tsne.png...
Computing t-SNE embedding
```

### Required libraries:

* keras, numpy, matplotlib, sklearn, h5py, pillow
