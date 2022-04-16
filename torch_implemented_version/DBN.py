import argparse

import numpy as np
import scipy.io as scio
from RBM import RBM
from RBM import lire_alpha_digit
from pathlib import Path
import torch
import matplotlib.pyplot as plt


from utils import visual_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ORI = Path(".")
experiment_path = ORI / "experiments" / "DBN"
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

class DBN():
    
    def __init__(self, p, q, hidden_dims, device=device):
        self.DBN = []
        hidden_dims_ = hidden_dims
        hidden_dims_.append(q)
        self.nb_layers = len(hidden_dims_)
        self.dim_input = p
        self.dim_output = q
        self.device = device
        

        # Construct DNN as chain of RBMs
        temp_dim_input = p
        for temp_dim_output in hidden_dims_:
            self.DBN.append(RBM(temp_dim_input, temp_dim_output, device=self.device))
            temp_dim_input = temp_dim_output
    
    def pretrain_DNN(self, X, nb_epoch, lr, batch_size):
        X_train = X.to(self.device)
        for i in range(self.nb_layers):
            print(f"Training layer {i}")
            self.DBN[i].fit(X_train, nb_epoch, lr, batch_size)
            X_train = self.DBN[i].entree_sortie_RBM(X_train)
            #print("\n")
    
    def gibbs(self, input, iterations):
        for n in range(iterations):
            init_ = input.to(self.device)
            for i in range(self.nb_layers):
                init_ = self.DBN[i].entree_sortie_RBM(init_)
            out_ = init_
            for i in range(self.nb_layers):
                idx_layer = self.nb_layers-i-1
                out_ = self.DBN[idx_layer].sortie_entree_RBM(out_)
        return out_
    
    def generer_image_DBN(self, nb_image, iterations):
        output = []
        for i in range(nb_image):
            input_init = torch.rand(1, self.dim_input).to(self.device)
            generated_image = self.gibbs(input_init, iterations)
            generated_image = generated_image.detach().cpu().numpy()
            generated_image =  np.reshape(generated_image, (20, 16))
            generated_image = np.round(generated_image)
            output.append(generated_image)
            
        print(f"Generated {nb_image} images.")
        return output


if __name__ == '__main__':
    args = arg_parse()
    mat_contents = scio.loadmat('./data/binaryalphadigs.mat')
    x = lire_alpha_digit(mat_contents['dat'], args.digit)
    x_tensor = torch.from_numpy(x).to(device).float() 
    epochs = args.iter
    lr = 0.1
    batch_size = 3
    dnn = DBN(320, 100, [100, 200, 200, 100] )
    dnn.pretrain_DNN(x_tensor, epochs, lr, batch_size)
    generated_images = dnn.generer_image_DBN(args.nb_img , 500)

    if args.show_img:
        visual_images(generated_images)

    if args.save_img:
        if experiment_path.is_dir() == False:
            experiment_path.mkdir(parents=True, exist_ok=True)
        for i in range(args.nb_img):
            plt.imsave( experiment_path / f"image_DBN-{epochs}-epochs-{i}.png", generated_images[i], cmap ='gray')
        print(f"Saved {args.nb_img} images")