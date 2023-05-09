import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

# Função para realizar a transformação gamma
def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Função para aplicar a negativa na imagem
def negative(image):
    return cv2.bitwise_not(image)

# Função para binarizar a imagem com um valor de limiar específico
def threshold(image, threshold_value):
    ret, thresh = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresh

# lista de nomes de arquivos de imagem
image_files = ['imagens/Fig0106(a)(bone-scan-GE).tif', 
              'imagens/Fig0106(b)(PET_image).tif',
              'imagens/Fig0115(a)(thum-print-loop).tif',
              'imagens/Fig0120(c)(ultrasound-tharoid structures).tif', 
              'imagens/Fig0117(b)(MRI-spine1-Vandy).tif']

# Loop pelas imagens no diretório
for image_file in image_files:
    # Carrega a imagem
    image = cv2.imread(image_file, 0)

    # Aplica a transformação gamma com um valor específico de gamma
    gamma = 0.5
    gamma_image = gamma_correction(image, gamma)

    # Aplica a negativa na imagem
    negative_image = negative(image)

    # Binariza a imagem com um valor específico de limiar
    threshold_value = 128
    threshold_image = threshold(image, threshold_value)

    # Mostra as imagens resultantes
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original")
    plt.subplot(1, 4, 2)
    plt.imshow(gamma_image, cmap="gray")
    plt.title("Gamma")
    plt.subplot(1, 4, 3)
    plt.imshow(negative_image, cmap="gray")
    plt.title("Negativa")
    plt.subplot(1, 4, 4)
    plt.imshow(threshold_image, cmap="gray")
    plt.title("Binarização")

    # Exibe os histogramas de cada imagem
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.hist(image.ravel(), 256, [0, 256])
    plt.title("Histograma Original")
    plt.subplot(1, 4, 2)
    plt.hist(gamma_image.ravel(), 256, [0, 256])
    plt.title("Histograma Gamma")
    plt.subplot(1, 4, 3)
    plt.hist(negative_image.ravel(), 256, [0, 256])
    plt.title("Histograma Negativa")
    plt.subplot(1, 4, 4)
    plt.hist(threshold_image.ravel(), 256, [0, 256])
    plt.title("Histograma Binarização")

    # Calcula a entropia de cada imagem
    entropy_image = cv2.calcHist([image], [0], None, [256], [0, 256])
    entropy_gamma_image = cv2.calcHist([gamma_image], [0], None, [256], [0, 256])
    entropy_negative_image = cv2.calcHist([negative_image], [0], None, [256], [0, 256])
    entropy_threshold_image = cv2.calcHist([threshold_image], [0], None, [256], [0, 256])
    entropy_image = cv2.normalize(entropy_image, entropy_image, norm_type=cv2.NORM_L1)
    entropy_gamma_image = cv2.normalize(entropy_gamma_image, entropy_gamma_image, norm_type=cv2.NORM_L1)
    entropy_negative_image = cv2.normalize(entropy_negative_image, entropy_negative_image, norm_type=cv2.NORM_L1)
    entropy_threshold_image = cv2.normalize(entropy_threshold_image, entropy_threshold_image, norm_type=cv2.NORM_L1)

    entropy_image = -np.sum(entropy_image * np.log2(entropy_image))
    entropy_gamma_image = -np.sum(entropy_gamma_image * np.log2(entropy_gamma_image))
    entropy_negative_image = -np.sum(entropy_negative_image * np.log2(entropy_negative_image))
    entropy_threshold_image = -np.sum(entropy_threshold_image * np.log2(entropy_threshold_image))

    print(f"Entropia da imagem original: {entropy_image}")
    print(f"Entropia da imagem com transformação gamma: {entropy_gamma_image}")
    print(f"Entropia da imagem negativa: {entropy_negative_image}")
    print(f"Entropia da imagem binarizada: {entropy_threshold_image}")

    plt.show()

