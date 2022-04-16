# Deep Learning 2 Course Project
This repo is the work for the project of course Deep Learning 2.

Implementation of RBM, DBN, DNN, VAE and experiment on Binary Alpha Digit and MNIST dataset. 

# Running RBM
Run RBM code with

``` python RBM.py --show_img ```

Optional arguments:
```
--digit: list of digits/letters that RBM structure will learn. Value of elements in range [0, 35]. Example: --digit 3 5 7. Default [3]

--iter: number of training iteration. Default 1000.

--nb_img: number of images to generate. Default 3.

--show_img: Mark to show images after generation.

--save_img: Mark to save generated images to /experiments/RBM
```
Example of running code with full optional arguments:

```python RBM.py --digit 6 10 --iter 800 --nb_img 5 --show_img --save_img```

# Running DBN

Run DBN code with

``` python DBN.py --show_img ```

Optional arguments: Same as RBM

Example of running code with full optional arguments:

```python DBN.py --digit 6 10 --iter 10 --nb_img 5 --show_img --save_img```
