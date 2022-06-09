# autoregressive-poisoning
Code for the paper [Autoregressive Perturbations for Data Poisoning](http://arxiv.org/abs/2206.03693) by Pedro Sandoval-Segura, Vasu Singla, Jonas Geiping, Micah Goldblum, Tom Goldstein, David W. Jacobs.

## Clean-up in progress!
This repo will be completely updated by 6/19/22.

### Try and train a model yourself!
We release our AR poisons as Zip files containing PNG images for easy viewing via [Google Drive](https://drive.google.com/drive/folders/1ze0cKAXNcPRkC0TMmObMj-g7Gspp1DpL?usp=sharing). This includes a
- CIFAR-10 AR Poison: ar-cifar-10.zip
- CIFAR-100 AR Poison: ar-cifar-100.zip
- SVHN AR Poison: 
- STL AR Poison: 

After unzipping, these poisons can be loaded using the `AdversarialPoison`, which subclasses `torch.utils.data.Dataset`. A model which trains on our AR poisons is unable to generalize to the (clean) test set.
