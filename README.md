# IJEPA-Enhanced

IJEPA-Enhanced aims to expand the usability of image JEPA models for real world usage. IJEPA-Enhanced employs a variety of techniques to improve the efficiency of the model for inference and training.

The main features are:
* variable resolution (NaViT, Pack n patch)
* token merging (TOME)

See a full writeup [here](https://theadamcolton.github.io/image-ssl-on-a-shoestring). I trained a ViT S for 6 hours on a RTX 3090 using the default config, it gets 27% validation accuracy on imagenet1k.


I trained a larger ViT-L for 160 hours on the same RTX 3090. It obtained 56% validation accuracy on imagenet1k.


### Variable Resolutions

Training batches are composed of images of different resolutions and aspect ratios. Image patches are packed together, and padded to form a batch of constant sequence length. 

Usually ViTs crop the image into a square. Center cropping is a destructive augmentation and probably hurts modelling performance for images with non-square aspect ratios. IJEPA-Enhanced allows you to choose the image resolution. Larger resolution images require more compute to be processed, but allow better predictions. Lower resolution images are faster and less accurate.

IJEPA-Enhanced uses factorized learnable positional embeddings. The original JEPA paper used sinusoidal non-factorized and non-learnable position embeddings. The position embeddings used in IJEPA-Enhanced are better for variable resolution training.

### Token merging

During training and inference, a bipartite graph selection algorithm is used to merge similar tokens, which reduces the sequence length with a minimal reduction in accuracy. This allows for more efficient memory usage during inference.

There have been some follow up papers citing ToME, for example [ToFu](https://openaccess.thecvf.com/content/WACV2024/papers/Kim_Token_Fusion_Bridging_the_Gap_Between_Token_Pruning_and_Token_WACV_2024_paper.pdf). IJEPA-Enhanced simply uses average merging for all layers, but potentially more merging techniques exist to be explored. 

Token merging in IJEPA-Enhanced is slight different than discribed in the ToME paper. Because each sequence of tokens in the batch can be composed of one or more images, IJEPA-Enhanced keeps track of which tokens are allowed to be merged. Padding tokens are allowed to be merged with other padding tokens. Tokens from an image are only allowed to be merged with other tokens from the same exact image. This logic is implemented in `ijepa_enhanced/tome.py`.

### Known bugs and issues

* The IJEPA paper selects the context region using a unit aspect ratio. This is unacceptable for IJEPA-Enhanced, because we want to train the encoder with variable aspect ratios. This change might effect training dynamics.
* IJEPA Enhanced uses padding less efficiently than Patch n pack. They report an average padding rate of 2%. In this project there is about a 10% padding rate. The higher padding is because context/prediction tokens are packed together.
* The imagenet@1 accuracy typically increases for the first 50k steps and then falters and decreases. This is as measured by a linear probe by training on mean pooled embeddings from the encoder for 50 epochs. The accuracy measured by finetuning the prediction similarly falters after about 50k steps. This decrease in evaluation performance could be due to representational collapse. Training for many more iterations never seems to result in better performance. I also noticed this decrease with the official IJEPA code. So the problem might be with the small ViT size.

