import numpy as np
import cv2
from scipy import fftpack
import matplotlib.pyplot as plt


# Função para aplicar um filtro de banda passa
def bandpass_filter(image, low_freq, high_freq):
    # Calcular a transformada de Fourier da imagem
    f_transform = fftpack.fft2(image)
    f_shift = fftpack.fftshift(f_transform)

    # Obter as dimensões da imagem
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Criar uma máscara de banda passa
    mask = np.zeros((rows, cols), dtype=np.uint8)
    mask[crow - low_freq:crow + low_freq, ccol - low_freq:ccol + low_freq] = 1
    mask[crow - high_freq:crow + high_freq, ccol - high_freq:ccol + high_freq] = 0

    # Aplicar a máscara à transformada de Fourier
    f_shift_filtered = f_shift * mask

    # Inverter a transformada de Fourier para obter a imagem filtrada
    f_ishift = fftpack.ifftshift(f_shift_filtered)
    image_filtered = fftpack.ifft2(f_ishift)
    image_filtered = np.abs(image_filtered)

    return image_filtered


# Analisar a imagem filtrada para detectar picos de frequência
def detect_moire(image_filtered):
    # Calcular a transformada de Fourier da imagem filtrada
    f_transform = fftpack.fft2(image_filtered)
    f_shift = fftpack.fftshift(f_transform)
    print('f_shift = ', f_shift)
    # Calcular o espectro de magnitude
    magnitude_spectrum = np.log(np.abs(f_shift))

    # Definir um limiar para detectar picos
    #original = 99
    threshold = np.percentile(magnitude_spectrum, 99)

    # Detectar picos
    peaks = (magnitude_spectrum > threshold).astype(np.uint8)

    return peaks


def testa_moire(jpeg_image, low_freq, high_freq):
    print ('jpeg_image = ', jpeg_image)
    # Carregar a imagem
    image = cv2.imread(jpeg_image, 0)  # Carregar em escala de cinza
    print ('image.shape = ', image.shape)
    # Definir frequências para varredura
    # low_freq = 10
    # high_freq = 50
    print('low_freq = ', low_freq, ' high_freq = ', high_freq)
    # Aplicar o filtro de banda passa
    filtered_image = bandpass_filter(image, low_freq, high_freq)

    # Mostrar a imagem original e a imagem filtrada
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Imagem Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(filtered_image, cmap='gray')
    plt.title('Imagem Filtrada'), plt.xticks([]), plt.yticks([])
    plt.show()

    # Detectar moiré na imagem filtrada
    moire_peaks = detect_moire(filtered_image)

    # Mostrar os picos detectados
  #  plt.imshow(moire_peaks, cmap='hot')
  #  plt.title('Picos de Moiré Detectados')
  #  plt.show()
