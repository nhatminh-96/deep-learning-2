import numpy as np
import matplotlib.pyplot as plt
from ast import arg
import scipy.io as scio
from tqdm import tqdm
import argparse
from pathlib import Path
import utils


import torch
import torch.nn as nn

ORI = Path(".")
experiment_path = ORI / "experiments" / "RBM"
data_path = ORI / 'data/binaryalphadigs.mat'

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--digit', nargs="+", help='which digit/letter to learn, must be a list', type=int, default=[3])
    parser.add_argument('--iter', help='number of iteration to train', type=int, default = 1000)
    parser.add_argument('--nb_img', help='number of images to generate', type=int, default = 3)
    parser.add_argument('--show_img', help='showing generated images', action ='store_true')
    parser.add_argument('--save_img', help='saving generated images', action ='store_true')
    args = parser.parse_args()
    return args


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def lire_alpha_digit(data, label):
    """
        Create data from matlab raw data file.

        ---
        Parameters:
            label : n_array (n, )
                    Number which to extract images
            data : matrix
                    Data set from matlab file
    """


    out = []
    for number in label:
        extracted = np.array([data[number][i].flatten() for i in range(len(data[number]))])
        out.append(extracted)

    return np.concatenate(out)

class RBM():

    """
    This class is our RBM structure
    ---
    Parameters:
        p : int (1,)
            The number of neurons in the visible layer

        q : int (1,)
            The number of neurons in the hidden layer

    """

    def __init__(self, p, q, device=device):
        self.device = device
        self.W = torch.rand(p, q).to(self.device).float() # random initialization with distribution N(0, 0.01)
        self.a = torch.zeros(1, p).to(self.device).float()
        self.b = torch.zeros(1, q).to(self.device).float()
    
    def entree_sortie_RBM(self, data):
        """
        This function compute the hidden layer given the visible layer.
        According to theory: P(h_j = 1 | v) = sigm(b_j + \sum{w_{i,j} v_i}).
        ---
        Parameters:
            RBM: class
                    Our RBM
            data:
                    Our data
        """
        # data_ = data.to(self.device)
        z = data @ self.W # + self.b.view(-1, self.W.shape[1])
        hidden = torch.sigmoid(z)
        return hidden

    def sortie_entree_RBM(self, data):

        """
        This function compute the visible layer given the hidden layer.
        According to theory: P(v_i = 1 | h) = sigm(a_i + \sum{w_{i,j} h_j}).
        ---
        Parameters:
            RBM: class
                    Our RBM
            data:
                    Our data

        """
        z = torch.mm(data, self.W.T) + self.a
        visible = torch.sigmoid(z)
        return visible
    
    def update(self, val_update):

        """
        This function will update our RBM with values in val_update

        ---
        Parameters :
            val : vector (3,)
                    Containing the values to be updated to W, b, a of our RBM

        """

        self.W += val_update[0]
        self.a += val_update[1]
        self.b += val_update[2]
        
        
    def fit(self, x, iteration, lr, batch_size):

        """
        This function will train our RBM model to correspoding input images x
        ---
        Parameters:
            RBM: class
                    Our RBM structure

            x: matrix (num_of_samples, p)
                    Input flatten image

            iteration: int (1, )
                    The number of iteration of training process

            lr: int (1, )
                    Learning rate

            batch_size: int (1,)
                    Batch size in training
        """

        p,q = self.W.shape
        n = x.shape[0]
        print(f"Training Started. {iteration} iterations.")
        for i in tqdm(range(iteration)):
            x_copy = torch.clone(x)
            # shuflle
            t = torch.rand(4, 2, 3, 3)
            idx = torch.randperm(x_copy.shape[0])
            x_copy = x_copy[idx].view(x_copy.size())

            for batch in range(0, n, batch_size):

                v_0 = x_copy[batch : min(batch + batch_size, n), :]
                nb_samples = len(v_0)
                h_0 = (torch.rand(nb_samples, q).to(self.device) < self.entree_sortie_RBM(v_0)).float()
                v_1 = (torch.rand(size = (nb_samples, p)).to(self.device) < self.sortie_entree_RBM(h_0)).float()
                # Gradient
                d_a = torch.sum(v_0 - v_1, axis = 0)
                d_b = torch.sum(self.entree_sortie_RBM(v_0) - self.entree_sortie_RBM(v_1), axis = 0)
                d_W = torch.mm(v_0.T, self.entree_sortie_RBM(v_0)) - torch.mm(v_1.T, self.entree_sortie_RBM(v_1))

                lr_ = lr/nb_samples
                self.update([lr_ * d_W, lr_ * d_a, lr_ *d_b])

            hidden = self.entree_sortie_RBM(x_copy)
            approximated_data = self.sortie_entree_RBM(hidden)
            # error = np.linalg.norm(x_copy - approximated_data, ord='fro')**2 / x_copy.size
        print('End Training')
        print("--------------")

    def generate_image(self, nb_images, nb_iteration):
        """
        From trained RBM, this function will generate images.

        ---
        Parameters:
            RBM :
                    Our RBM class with the weights already trained and updated
            nb_images : int (1,)
                    The number of samples that we want to generate

            nb_iter : int (1,)
                    The number of iterations used during the generation

        """
        imgs = []
        p, q = self.W.shape
        for i in range(nb_images):
            input = (torch.rand(1, p).to(self.device) < 0.5).float()

            for iter in range(nb_iteration):
                h = (torch.rand(1, q).to(self.device) < self.entree_sortie_RBM(input)).float()
                input = (torch.rand(1, p).to(self.device) < self.sortie_entree_RBM(h)).float()

            output =  np.reshape(input.detach().cpu().numpy(), (20, 16))
            imgs.append(output)
        print(f"Generated {nb_images} images")
        return imgs

    def display(self):
        print(self.W, self.a, self.b)

if __name__ == '__main__':
    args = arg_parse()
    mat_contents = scio.loadmat(data_path)
    x = lire_alpha_digit(mat_contents['dat'], args.digit)
    x = torch.from_numpy(x).to(device).float()
    epochs = args.iter
    rbm = RBM(320, 200)
    #x = torch.rand(25, 320).to(device).float()
    lr = 0.1
    batch_size = 5
    rbm.fit(x, epochs, lr, batch_size)

    generated_images = rbm.generate_image(3 ,  12)
    
    # if args.show_img:
    #     utils.visual_images(generated_images)

    if args.save_img:
        if experiment_path.is_dir() == False:
            experiment_path.mkdir(parents=True, exist_ok=True)
        for i in range(args.nb_img):
            plt.imsave( experiment_path / f"image_RBM-{epochs}-epochs-{i}.png", generated_images[i], cmap ='gray')
        print(f"Saved {args.nb_img} images")