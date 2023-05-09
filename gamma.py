import numpy as np
import cv2
from matplotlib import pyplot as plt

# lista de nomes de arquivos de imagem
file_names = ['imagens/Fig0106(a)(bone-scan-GE).tif', 
              'imagens/Fig0106(b)(PET_image).tif',
              'imagens/Fig0115(a)(thum-print-loop).tif',
              'imagens/Fig0120(c)(ultrasound-tharoid structures).tif', 
              'imagens/Fig0117(b)(MRI-spine1-Vandy).tif']

# constante gamma
gamma = 0.5

# loop através de cada imagem
for file_name in file_names:
    # carregando a imagem
    img = cv2.imread(file_name, 0)
    
    # normalizando a imagem
    img_norm = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    
    # aplicando a transformação gamma
    img_gamma = np.power(img_norm, gamma)
    
    # normalizando a imagem novamente
    img_gamma = cv2.normalize(img_gamma, None, 0, 255, cv2.NORM_MINMAX)
    
    # convertendo a imagem para uint8
    img_gamma = img_gamma.astype('uint8')
    
    # exibindo as imagens
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Imagem Original')
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(img_gamma, cmap='gray')
    plt.title('Imagem com Transformação Gamma')
    plt.xticks([]), plt.yticks([])

    plt.show()
