# deep-learning-2
This repo is the work for  the project of course Deep Learning 2.

Implementation of RBM, DBN, DNN, VAE.

# Running RBM
Run RBM code with

python RBM.py 

optional arguments:

--digit: list of digits/letters that RBM structure will learn. Values of elements in range [0, 35]. Example: --digit 3 5 7. Default [3]

--iter: number of training iteration. Default 1000.

--nb_img: number of images to generate. Default 3.

Example of running code with full optional arguments:

python RBM.py --digit 6 10 --iter 800 --nb_img 5
