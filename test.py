from scipy import signal
from scipy import ndimage
from PIL import Image
import numpy as np


def image2arr(path: str):
    img = Image.open(path)
    img.load()
    img_array = np.asarray(img, dtype='int32')
    return img_array


def arr2image(path: str, img_arr: np.array):
    img_arr = (img_arr).astype(np.uint8)
    img = Image.fromarray(img_arr)
    img.save(path)
    print('saved to: ' + str(path))


def d(img_arr: np.array, n: int):
    img_red = img_arr[..., 0]
    img_green = img_arr[..., 1]
    img_blue = img_arr[..., 2]
    img_red = img_red[::n, ::n]
    img_green = img_green[::n, ::n]
    img_blue = img_blue[::n, ::n]
    ans_arr = np.stack((img_red, img_green, img_blue), axis=-1)
    print(img_red.shape)
    print(img_arr.shape)
    print(ans_arr.shape)
    return ans_arr


def u(img_arr: np.array, n: int):
    img_red = img_arr[..., 0]
    img_green = img_arr[..., 1]
    img_blue = img_arr[..., 2]
    img_red = img_red.repeat(n, axis=0).repeat(n, axis=1)
    img_green = img_green.repeat(n, axis=0).repeat(n, axis=1)
    img_blue = img_blue.repeat(n, axis=0).repeat(n, axis=1)
    ans_arr = np.stack((img_red, img_green, img_blue), axis=-1)
    print(img_red.shape)
    print(img_arr.shape)
    print(ans_arr.shape)
    return ans_arr


def de_noise(img_arr: np.array, n:int):


    # # Provided code:
    # shape = img_arr.shape
    # n_rows, n_cols = shape[0], shape[1]
    # # Code for part 4.a:
    # ba = [0, 0.1, 0.2, 0.5, 0.6, 1]
    # da = [1, 0, 1]
    # fir = signal.remez(25, ba, da, fs=2)
    # w = signal.windows.gaussian(99, 5.0)
    img_red = img_arr[..., 0]
    img_green = img_arr[..., 1]
    img_blue = img_arr[..., 2]
    img_red = ndimage.median_filter(img_red,3)
    img_green = ndimage.median_filter(img_green, 3)
    img_blue = ndimage.median_filter(img_blue, 3)

    # img_red = signal.sepfir2d(img_red, w, w)
    # img_green = signal.sepfir2d(img_green, w, w)
    # img_blue = signal.sepfir2d(img_blue, w, w)

    # for i in range(0, n_rows):
    #     img_red[i, :] = signal.convolve(img_red[i, :], fir, 'same')
    #
    # for j in range(0, n_cols):
    #     img_red[:, j] = signal.convolve(img_red[:, j], fir, 'same')
    #
    # for i in range(0, n_rows):
    #     img_green[i, :] = signal.convolve(img_green[i, :], fir, 'same')
    #
    # for j in range(0, n_cols):
    #     img_green[:, j] = signal.convolve(img_green[:, j], fir, 'same')
    #
    # for i in range(0, n_rows):
    #     img_blue[i, :] = signal.convolve(img_blue[i, :], fir, 'same')
    #
    # for j in range(0, n_cols):
    #     img_blue[:, j] = signal.convolve(img_blue[:, j], fir, 'same')

    # img_red_fft = np.fft.rfft2(img_red)
    # img_green_fft = np.fft.rfft2(img_green)
    # img_blue_fft = np.fft.rfft2(img_blue)
    # nr = 0.000001 * abs(np.max(img_red_fft))
    # ng = 0.000001 * abs(np.max(img_green_fft))
    # nb = 0.000001 * abs(np.max(img_blue_fft))

    # rows, cols = img_red_fft.shape
    # for i in range(0, rows):
    #     for j in range(0, cols):
    #         if abs(img_red_fft[i,j]) < nr:
    #             img_red_fft[i,j] = 0
    # rows, cols = img_green_fft.shape
    # for i in range(0, rows):
    #     for j in range(0, cols):
    #         if abs(img_green_fft[i,j]) < ng:
    #             img_green_fft[i,j] = 0
    # rows, cols = img_blue_fft.shape
    # for i in range(0, rows):
    #     for j in range(0, cols):
    #         if abs(img_blue_fft[i,j]) < nb:
    #             img_blue_fft[i,j] = 0
    # img_red = np.fft.irfft2(img_red_fft)
    # img_green = np.fft.irfft2(img_green_fft)
    # img_blue = np.fft.irfft2(img_blue_fft)
    ans_arr = np.stack((img_red, img_green, img_blue), axis=-1)

    return ans_arr


if __name__ == "__main__":

    TEST_IMAGE_ADDR = 'noise.jpg'
    IMG_ARR = image2arr(TEST_IMAGE_ADDR)
    arr2image('original.jpg', IMG_ARR)
    ANS_ARR = u(IMG_ARR, 3)
    arr2image('upsample.jpg', ANS_ARR)
    ANS_ARR = d(IMG_ARR, 3)
    arr2image('downsample.jpg', ANS_ARR)
    ANS_ARR = de_noise(IMG_ARR, 10)
    arr2image('denoised.jpg', ANS_ARR)



