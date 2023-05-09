import numpy as np
import cv2
from matplotlib import pyplot as plt

# lista de nomes de arquivos de imagem
file_names = ['imagens/Fig0106(a)(bone-scan-GE).tif', 
              'imagens/Fig0106(b)(PET_image).tif',
              'imagens/Fig0115(a)(thum-print-loop).tif',
              'imagens/Fig0120(c)(ultrasound-tharoid structures).tif', 
              'imagens/Fig0117(b)(MRI-spine1-Vandy).tif']

# constante de limiar para binarização
threshold = 0.5

# loop através de cada imagem
for file_name in file_names:
    # carregando a imagem
    img = cv2.imread(file_name, 0)
    
    # aplicando a negativa
    img_negative = 255 - img
    
    # normalizando a imagem
    img_norm = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    
    # aplicando a binarização
    img_binary = np.where(img_norm > threshold, 255, 0)
    
    # exibindo as imagens
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Imagem Original')
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 3, 2)
    plt.imshow(img_negative, cmap='gray')
    plt.title('Imagem Negativa')
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 3, 3)
    plt.imshow(img_binary, cmap='gray')
    plt.title('Imagem Binarizada')
    plt.xticks([]), plt.yticks([])

    plt.show()
