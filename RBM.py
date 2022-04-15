import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from tqdm import tqdm
import argparse
from PIL import Image

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--digit', nargs="+", help='which digit/letter to learn, must be a list', type=int, default=[3])
    parser.add_argument('--iter', help='number of iteration to train', type=int, default = 1000)
    parser.add_argument('--nb_img', help='number of images to generate', type=int, default = 3)
    args = parser.parse_args()
    return args



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

    def __init__(self, p, q):
        self.W = np.random.normal(0, 0.01, (p, q)) # random initialization with distribution N(0, 0.01)
        self.a = np.zeros(p)
        self.b = np.zeros(q)


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

        hidden = 1. / (1. + np.exp(-self.b.reshape(1,-1) - data @ self.W ))

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

        visible = 1. / (1. + np.exp(- self.a.reshape(1,-1) - data @ self.W.T))

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
            x_copy = x.copy()
            np.random.shuffle(x_copy)

            for batch in range(0, n, batch_size):

                v_0 = x_copy[batch : min(batch + batch_size, n), :]
                h_0 = (np.random.uniform(size = (len(v_0), q)) < self.entree_sortie_RBM(v_0)).astype('float')
                v_1 = (np.random.uniform(size = (len(v_0), p)) < self.sortie_entree_RBM(h_0)).astype('float')
                # Gradient
                d_a = np.sum(v_0 - v_1, axis = 0)
                d_b = np.sum(self.entree_sortie_RBM(v_0) - self.entree_sortie_RBM(v_1), axis = 0)
                d_W = np.dot(v_0.T, self.entree_sortie_RBM(v_0)) - np.dot(v_1.T, self.entree_sortie_RBM(v_1))

                lr_ = lr/len(v_0)
                self.update([lr_ * d_W, lr_ * d_a, lr_ *d_b])

            hidden = self.entree_sortie_RBM(x_copy)
            approximated_data = self.sortie_entree_RBM(hidden)
            error = np.linalg.norm(x_copy - approximated_data, ord='fro')**2 / x_copy.size
        print('End Training')


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

        p,q = self.W.shape
        for i in range(nb_images):
            input = (np.random.uniform(size=p) < 0.5).astype('float')

            for iter in range(nb_iteration):
                h = (np.random.uniform(size = q) < self.entree_sortie_RBM(input)).astype('float')
                input = (np.random.uniform(size = p) < self.sortie_entree_RBM(h)).astype('float')

            output =  np.reshape(input, (20, 16))
            plt.imshow(output, cmap = 'gray')
            plt.title(f'Generated images {i+1}')
            plt.show()

            #plt.imsave(f"image-{i}.png", output, cmap ='gray')
            #plt.close()
        print(f"Generated {nb_images} images")

    def display(self):
        print(self.W, self.a, self.b)


if __name__ == '__main__':
    args = arg_parse()
    mat_contents = scio.loadmat('./data/binaryalphadigs.mat')
    x = lire_alpha_digit(mat_contents['dat'], args.digit)
    iteration = args.iter
    lr = 0.1
    batch_size = 3
    rbm = RBM(320, 200)
    rbm.fit( x, iteration, lr, batch_size)
    rbm.generate_image(args.nb_img ,  500)
