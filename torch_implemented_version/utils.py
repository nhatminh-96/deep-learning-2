import matplotlib.pyplot as plt
import numpy as np

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



def visual_images(list_images):
    nb_imgs = len(list_images)
    # 5 images each column
    nb_columns = 5 if nb_imgs >= 5 else nb_imgs
    nb_rows = nb_imgs//5 + 1 if nb_imgs%5 != 0 else nb_imgs//5
    fig, axs = plt.subplots(nb_rows, nb_columns, figsize=(2*nb_columns, 2*nb_rows))
    for image, ax in zip(list_images, axs.flatten()):
        ax.imshow(image, cmap='gray')
        ax.axis('off')