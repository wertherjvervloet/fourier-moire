# This is a sample Python script.
import logging
import os
import numpy as np
import cv2
from scipy import fftpack
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
from filtropassabanda import testa_moire

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

imagePath = 'imagens/'
imagem = 'imagens\L1001283.jpg'
imagem = 'imagens\m20160420.jpg'
imagem = 'imagens\m20160728.jpg'
imagem = 'imagens\m20171019.jpg'
imagem = 'imagens\leque.jpg'

try:
    imageFiles = [f for f in os.listdir(imagePath) if os.path.isfile(os.path.join(imagePath, f))]
    print(imageFiles)

    logging.info(f'Start processing {len(imageFiles)} images in {imagePath}')
    '''
    # Configurando a barra de progresso com descrição e comportamento após conclusão
    with tqdm(total=len(imageFiles), desc="Processing Images", leave=False, ncols=100) as pbar:
        for result in pool.imap_unordered(process_image, [(f, imagePath, trainFolderPath) for f in imageFiles]):
            print("result = ", result)
            results.append(result)
            pbar.update()  # Atualiza a barra de progresso
            # Monitorar a utilização da CPU a cada 10 iterações para reduzir o número de logs
            if len(results) % 10 == 0:
                cpu_usage = psutil.cpu_percent(interval=None, percpu=True)
                logging.info(f'CPU Usage: {cpu_usage}')

    pool.close()
    pool.join()

    logging.info(f'Finished processing images in {imagePath}')
    print("results = ", results)
    return sum(results)
    
    '''
    for img in imageFiles:
        image = cv2.imread(imagePath+img, 0)  # Carregar em escala de cinza

        # Calcular a Transformada de Fourier
        f_transform = fftpack.fft2(image)
        f_shift = fftpack.fftshift(f_transform)


        # Calcular o espectro de magnitude
        magnitude_spectrum = np.log(np.abs(f_shift))
        #newmag = sigmoid(magnitude_spectrum)
        newmag = magnitude_spectrum
      #  newmag = nn.Sigmoid(magnitude_spectrum)

        # Mostrar a imagem original e o espectro de magnitude
        plt.subplot(121), plt.imshow(image, cmap='gray')
        plt.title('Imagem Original'), plt.xticks([]), plt.yticks([])
    #   plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    #   plt.title('Espectro de Magnitude'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(newmag, cmap='gray')
        plt.title('Espectro de Magnitude'), plt.xticks([]), plt.yticks([])
        plt.show()
    exit(0)

    testa_moire(imagem, 00, 240)
except Exception as e:
    logging.error(f'erro no try: {e}')
    exit (0)


