import numpy as np
import scipy.io as scio
from RBM import *
from pathlib import Path
#from tf.keras.datasets import mnist
from sklearn.metrics import accuracy_score

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


ORI = Path(".")

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--digit', nargs="+", help='which digit/letter to learn, must be a list', type=int, default=[3])
    parser.add_argument('--iter', help='number of iteration to train', type=int, default = 1000)
    parser.add_argument('--nb_img', help='number of images to generate', type=int, default = 3)
    parser.add_argument('--show_img', help='showing generated images', action ='store_true')
    parser.add_argument('--save_img', help='saving generated images', action ='store_true')
    args = parser.parse_args()
    return args

class DNN():
    
    def __init__(self, p, q, hidden_dims):
        self.DNN = []
        hidden_dims_ = hidden_dims
        #hidden_dims_.append(q)
        self.nb_layers = len(hidden_dims_)
        self.dim_input = p
        self.dim_output = q

        # Construct DNN as chain of DBMs
        temp_dim_input = p
        for temp_dim_output in hidden_dims_:
            self.DNN.append(RBM(temp_dim_input, temp_dim_output))
            temp_dim_input = temp_dim_output
        
        self.classification_head = RBM(temp_dim_input, q)
    
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
    
    def generer_image_DBN(self, nb_image, iterations):
        output = []
        for i in range(nb_image):
            input_init = np.random.rand(1, self.dim_input)
            generated_image = self.gibbs(input_init, iterations)
            generated_image =  np.reshape(generated_image, (20, 16))
            generated_image = np.round(generated_image)
            output.append(generated_image)
            
        print(f"Generated {nb_image} images.")
        return output

def calcul_softmax(RBM_, input):
        #W = torch.from_numpy(RBM_.W).to(device)
        #b = torch.from_numpy(RBM_.b).to(device)
        W = RBM_.W
        b = RBM_.b
        #input = torch.from_numpy(input)
        #print(W.shape)
        #print(input.shape)
        #print(b.shape)
        RBM_out = torch.matmul(input, W) + b
        
        #RBM_out = input @ W + b
        #print(RBM_out.shape)
        #temp = torch.exp(RBM_out- torch.max(RBM_out, axis=1)[0][:,None])
        temp = torch.exp(RBM_out)
        temp_sum = torch.sum(temp, axis=1)
        softmax = temp/temp_sum[:, None]
        return softmax

def entre_sortie_reseau(DNN_, input):
    output = []
    input_ = input
    for layer in DNN_.DNN:
        input_ = layer.entree_sortie_RBM(input_)
        output.append(input_)
    # Last layer - softmax classification head
    #input_ = torch.from_numpy(input_).to(device)
    softmax_proba = calcul_softmax(DNN_.classification_head, input_)
    output.append(softmax_proba)
    return output

def retropropagation(DNN_, input, label, epochs, lr, batch_size):
    tracking = []
    input = torch.from_numpy(input).to(device).float()
    label = torch.from_numpy(label).to(device).float()
    with tqdm(total = epochs, unit_scale=True, postfix={'loss ':0.0}, ncols=100) as pbar:
        for epoch in range(epochs):
            #idx_ = np.random.permutation(input.shape[0])
            idx_ = torch.randperm(input.shape[0])
            input = input[idx_]
            label = label[idx_]
            for i in range(0, input.shape[0] - batch_size, batch_size):
                input_batch = input[i:(i+batch_size),]
                label_batch = label[i:(i+batch_size),]
                out_DNN = entre_sortie_reseau(DNN_, input_batch)
                y_hat = out_DNN[-1]
                #print(y_hat)
        
                gradients_ = {}
        
                dxi_1 = y_hat - label_batch
                x_i_1 = out_DNN[-2]

                #x_i_1 = torch.from_numpy(x_i_1).to(device)
                gradients_['W_last'] = torch.matmul(x_i_1.T, dxi_1)
                gradients_['b_last'] = torch.mean(dxi_1, axis=0)
        
                dxi1 = dxi_1
                for j in range(DNN_.nb_layers):
        
                    idx = DNN_.nb_layers - j - 1
                    x_i = out_DNN[idx]
                    if idx == 0:
                        x_i_1 = input_batch
                    else:
                        x_i_1 = out_DNN[idx-1]
        
        
                    if idx != DNN_.nb_layers -1:
                        W_i1 = DNN_.DNN[idx+1].W
                    else:
                        W_i1 = DNN_.classification_head.W
                    # W_i1 = torch.from_numpy(W_i1).to(device)
                    # x_i = torch.from_numpy(x_i).to(device)
                    multi = torch.mul(x_i, (1-x_i))
                    dx_i = torch.mul(torch.matmul(dxi1, W_i1.T), multi)
                    dxi1 = dx_i

                    #x_i_1 = torch.from_numpy(x_i_1).to(device)
                    #dx_i = dx_i.to('cpu').detach().numpy()
                    #print(dx_i.shape)
                    #print(x_i_1.shape)
                    #mat_product = x_i_1.T @ dx_i
                    #dx_i = torch.from_numpy(dx_i).to(device)
                    #mat_product = torch.from_numpy(mat_product).to(device)
                    gradients_['W_'+str(idx)] = x_i_1.T @ dx_i
                    gradients_['b_'+str(idx)] = torch.mean(dx_i, axis=0)
                #print(gradients_['W_last'].shape)
                DNN_.classification_head.W = DNN_.classification_head.W - lr*gradients_['W_last']
                DNN_.classification_head.b = DNN_.classification_head.b - lr*gradients_['b_last']
                for k in range(DNN_.nb_layers):
                    DNN_.DNN[k].W = DNN_.DNN[k].W - lr*gradients_['W_'+str(k)]
                    DNN_.DNN[k].b = DNN_.DNN[k].b - lr*gradients_['b_'+str(k)]
    
            y_hat = entre_sortie_reseau(DNN_, input)[-1]
            #print(y_hat)
            l = -torch.sum(label*torch.log(y_hat))/y_hat.shape[0]
            loss_cpu = l.to('cpu').detach().numpy()
            pbar.set_postfix({'loss ':f"{loss_cpu:.5f}"})
            pbar.update(1)
            tracking.append(l)
            #print(f'The Loss at Epoch : {epoch} is {l}')
    tracking = [tracking[i].to('cpu').detach().numpy() for i in range(len(tracking))]
    return DNN_, tracking

def test_DNN(DNN_, X, Y):
    X = torch.from_numpy(X).to(device).float()
    out = entre_sortie_reseau(DNN_, X)
    y_hat = out[-1]
    y_hat = y_hat.to('cpu').detach().numpy()
    loss  = -np.sum(Y*np.log(y_hat+1e-9))/y_hat.shape[0]
    y_hat = (y_hat > 0.5)*1
    accuracy = accuracy_score(Y, y_hat)
    print(f'We have an accuracy of {np.round(accuracy*100,3)}% \n and a entropy loss of {np.round(loss,3)}')
    return loss, accuracy