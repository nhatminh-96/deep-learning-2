import numpy as np
import scipy.io as scio
from RBM import *
from sklearn.metrics import accuracy_score

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--digit', nargs="+", help='which digit/letter to learn, must be a list', type=int, default=[3])
    parser.add_argument('--iter', help='number of iteration to train', type=int, default = 1000)
    parser.add_argument('--nb_img', help='number of images to generate', type=int, default = 3)
    args = parser.parse_args()
    return args

class DNN():
    
    def __init__(self, p, q, hidden_dims):
        self.DNN = []
        hidden_dims_ = hidden_dims
        hidden_dims_.append(q)
        self.nb_layers = len(hidden_dims_)
        self.dim_input = p
        self.dim_output = q

        # Construct DNN as chain of DBMs
        temp_dim_input = p
        for temp_dim_output in hidden_dims_:
            self.DNN.append(RBM(temp_dim_input, temp_dim_output))
            temp_dim_input = temp_dim_output
    
    def pretrain_DNN(self, X, nb_epoch, lr, batch_size):
        X_train = X
        for i in range(self.nb_layers):
            print(f"Training layer {i}")
            self.DNN[i].fit(X_train, nb_epoch, lr, batch_size)
            X_train = self.DNN[i].entree_sortie_RBM(X_train)
            #print("\n")
    
    def gibbs(self, input, iterations):
        for n in range(iterations):
            init_ = input
            for i in range(self.nb_layers):
                init_ = self.DNN[i].entree_sortie_RBM(init_)
            out_ = init_
            for i in range(self.nb_layers):
                idx_layer = self.nb_layers-i-1
                out_ = self.DNN[idx_layer].sortie_entree_RBM(out_)
        return out_
    
    def generer_image_DBN(self, nb_image, iterations, nb_epochs):
        output = []
        for i in range(nb_image):
            input_init = np.random.rand(1, self.dim_input)
            generated_image = self.gibbs(input_init, iterations)
            generated_image =  np.reshape(generated_image, (20, 16))
            generated_image = np.round(generated_image)
            output.append(generated_image)
            plt.imsave(f"image_dbn-{nb_epochs}-epochs-{i}.png", generated_image, cmap ='gray')
        print(f"Generated {nb_image} images.")
        return output


if __name__ == '__main__':
    args = arg_parse()
    mat_contents = scio.loadmat('./data/binaryalphadigs.mat')
    x = lire_alpha_digit(mat_contents['dat'], args.digit)
    epochs = args.iter
    lr = 0.1
    batch_size = 3
    dnn = DNN(320, 100, [100, 200, 200, 100] )
    dnn.pretrain_DNN(x, epochs, lr, batch_size)
    generated = dnn.generer_image_DBN(args.nb_img , 500, epochs)