# IJEPA-Enhanced

IJEPA-Enhanced aims to expand the usability of image JEPA models for real world usage. IJEPA-Enhanced employs a variety of cutting edge techniques for improving ViT efficiency and performance.

* variable resolution (NaViT, Pack n patch)
* token merging (TOME)
* discrete embedding space
* null attention

### Variable Resolutions

Training batches are composed of images of different resolutions and aspect ratios. Image patches are packed together, and possibly padded. IJEPA-Enhanced uses factorized learnable positional embeddings. 

Usually ViTs crop the image into a square. Center cropping is a destructive augmentation probably hurts modelling performance for images with non-square aspect ratios. IJEPA-Enhanced allows you to choose the image resolution. Larger resolution images require more compute to be processed, but allow better predictions. Lower resolution images are faster and less accurate.

### Token merging

During training and inference, a bipartite graph selection algorithm is used to merge similar tokens, which reduces the sequence length with a minimal reduction in accuracy. This allows for more efficient memory usage during inference.

### Discrete embedding space

The original IJEPA used a patch-wise vector embeddings in real space. They used the L1 distance metric to measure patch prediction loss. IJEPA-Enhanced converts each output vector of the encoder into a sequence of discrete codes. The predictor produces multiheaded logits for each patch to be predicted. NLL is used instead of L1 loss. 

The discrete embedding space is used in order to reduce the space requirements of storing many image embeddings. Real space embeddings can be quickly decoded from the discrete codes.

### Null attention

ViTs can benifit from allowing the attention process to attend to nothing. This surmounts to adding a 1 to the denominator of the softmax normalization of the self attention matrices.

### Bugs

* Patchnpack context-target the target blocks can be sampled exterior to the context rect which is not as described in the ijepa paper

### TODO

* During training, report rate-distortion loss using nd-ngram terms
* Report prediction-sequence padding rate, context padding rate, and target padding rate
* Make transformer model the same as used in IJEPA
* Load pretrained IJEPA models
* LFQ and Masked cross entropy loss, hard CE loss VS soft CE loss, (discretizing labels vs leaving labels as normalized probs)
* should position information be given for tokens that are not being predicted? Yes: They will be stored as nd-kgrams with their positions explicitly known, they don't need to encode their positions in and of themselves.
* Test that attention mask works
* Benchmark patchnpack and optimize
* TOME
