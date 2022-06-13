# autoregressive-poisoning
Code for the paper [Autoregressive Perturbations for Data Poisoning](http://arxiv.org/abs/2206.03693) by Pedro Sandoval-Segura, Vasu Singla, Jonas Geiping, Micah Goldblum, Tom Goldstein, David W. Jacobs.

<div align="center">
  <img width="95%" alt="RandomBatchOfARPoison" src="imgs/appendix-batches-each-poison.006.png">
</div>
<div align="center">
  A random batch of 30 images and the corresponding normalized perturbation from our AR CIFAR-10 Poison.
</div>

## Clean-up in progress!
This repo will be completely updated by 6/19/22.

## Finding AR process coefficients

To find a set of 10 AR processes, run:

```
python autoregressive_param_finder.py --total=10 --required_nm_response=10 --gen_norm_upper_bound=50
```

This command will save a file named `params-classes-10-mr-10.pt` using `torch.save`. The format will be identical to that of `RANDOM_3C_AR_PARAMS_RNMR_10` within autoregressive_params.py, a list of `torch.tensor`. Additional information can be found in Appendix A.3.

## Generating AR perturbations

See **notebooks/Generate-AR-Perturbations-from-Coefficients.ipynb** for an example of how to load AR coefficients and generate an AR perturbation of a given size and norm.

## Training a model on a poison



### Try and train a model yourself!
We release our AR poisons as Zip files containing PNG images for easy viewing via [Google Drive](https://drive.google.com/drive/folders/1ze0cKAXNcPRkC0TMmObMj-g7Gspp1DpL?usp=sharing). This includes a
- CIFAR-10 AR Poison: ar-cifar-10.zip
- CIFAR-100 AR Poison: ar-cifar-100.zip
- SVHN AR Poison: ar-svhn.zip
- STL AR Poison: ar-stl.zip

After unzipping, these poisons can be loaded using `AdversarialPoison`, a subclass of `torch.utils.data.Dataset`. A model which trains on our AR poisons is unable to generalize to the (clean) test set.
