from flask import Flask, redirect, render_template, url_for, request, flash, session
from werkzeug.utils import secure_filename
import os

UPLOAD_FOLDER = os.path.join("static", "imagestorage")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "yes"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def welcome():
    return render_template("index.html")

@app.route("/image", methods=["GET","POST"])
def imageinput():
    if request.method == "GET":
        return render_template("ImageProcessing.html")
    else:
        way = str(request.form["how"])
        if 'file' not in request.files:
            flash('No file part')
            return redirect()
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect()
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            session["path"] = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            return redirect(url_for('imageprocess', how = way))
        return redirect(url_for("imageprocess", how = way))



# @app.route("/audio/<how>")
# def audio(how):
#     #audfile = session["file"]
#     return f"{how} audio processing"

@app.route("/imageprocess/<how>")
def imageprocess(how):
    imgfile = session["path"]
    flash("Just a moment!")
    para = 5
    imgarr = image2arr(imgfile)
    if how == "Image Optimization":
        after = enhance(imgarr)
    elif how == "Image Compression":
        after = d(imgarr,para)
    elif how == "Denoising (Gaussian)":
        after = de_noise_gaussian(imgarr,para)
    elif how == "Denoising (Median)":
        after = de_noise_median(imgarr)
    elif how == "Edge detection":
        after = edge(imgarr)
    arr2image(app.config['UPLOAD_FOLDER'] + "/after.jpg", after)
    imgfile = os.path.join(app.config['UPLOAD_FOLDER'], "after.jpg")
    return render_template("out.html", image = "../" + imgfile)

@app.route("/team")
def team():
    return render_template("Contact.html")



from scipy import signal
from scipy import ndimage
from PIL import Image, ImageFilter
import numpy as np


def image2arr(path: str):
    img = Image.open(path)
    img.load()
    img_array = np.asarray(img, dtype='int32')
    return img_array


def arr2image(path: str, img_arr: np.array):
    img_arr = img_arr.astype(np.uint8)
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
    return ans_arr


def u(img_arr: np.array, n: int):
    img_red = img_arr[..., 0]
    img_green = img_arr[..., 1]
    img_blue = img_arr[..., 2]
    img_red = img_red.repeat(n, axis=0).repeat(n, axis=1)
    img_green = img_green.repeat(n, axis=0).repeat(n, axis=1)
    img_blue = img_blue.repeat(n, axis=0).repeat(n, axis=1)
    ans_arr = np.stack((img_red, img_green, img_blue), axis=-1)
    return ans_arr


def de_noise_median(img_arr: np.array):
    img_red = img_arr[..., 0]
    img_green = img_arr[..., 1]
    img_blue = img_arr[..., 2]
    img_red = ndimage.median_filter(img_red, 3)
    img_green = ndimage.median_filter(img_green, 3)
    img_blue = ndimage.median_filter(img_blue, 3)
    ans_arr = np.stack((img_red, img_green, img_blue), axis=-1)
    return ans_arr


def de_noise_gaussian(img_arr: np.array, n: int):
    img_red = img_arr[..., 0]
    img_green = img_arr[..., 1]
    img_blue = img_arr[..., 2]
    img_red = ndimage.gaussian_filter(img_red, n)
    img_green = ndimage.gaussian_filter(img_green, n)
    img_blue = ndimage.gaussian_filter(img_blue, n)
    ans_arr = np.stack((img_red, img_green, img_blue), axis=-1)
    return ans_arr


def enhance(img_arr: np.array):
    oprand = de_noise_gaussian(d(de_noise_gaussian(u(img_arr, 4), 2), 2), 1)
    return oprand


def sharpen(img_arr: np.array):
    alpha = 0.01
    return img_arr + alpha * (img_arr - de_noise_gaussian(img_arr, 1))


def edge(img_arr: np.array):
    img_red = img_arr[..., 0]
    img_green = img_arr[..., 1]
    img_blue = img_arr[..., 2]

    img_arr = (img_red + img_blue + img_green) / 3

    h = [1, -1]

    a_arr = np.zeros(img_arr.shape)
    for n in range(img_arr.shape[0]):
        a_arr[n, :] = signal.convolve(img_arr[n, :], h, 'same')

    b_arr = np.zeros(img_arr.shape)
    for n in range(img_arr.shape[1]):
        b_arr[:, n] = signal.convolve(img_arr[:, n], h, 'same')

    ans_arr = np.zeros(img_arr.shape)
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            ans_arr[i, j] = ((a_arr[i, j]) ** 2 + (b_arr[i, j]) ** 2) ** 0.5

    return ans_arr


def invert(img_arr: np.array):
    ones = np.ones_like(img_arr) * 235
    return ones - img_arr


if __name__ == "__main__":
    TEST_IMAGE_ADDR = 'pictures/hamster.jpg'
    IMG_ARR = image2arr(TEST_IMAGE_ADDR)
    arr2image('pictures/original.jpg', IMG_ARR)
    ANS_ARR = u(IMG_ARR, 3)
    arr2image('pictures/upsample.jpg', ANS_ARR)
    ANS_ARR = d(IMG_ARR, 3)
    arr2image('pictures/downsample.jpg', ANS_ARR)
    ANS_ARR = de_noise_median(IMG_ARR)
    arr2image('pictures/median.jpg', ANS_ARR)
    ANS_ARR = de_noise_gaussian(IMG_ARR, 2)
    arr2image('pictures/gauss.jpg', ANS_ARR)
    ANS_ARR = enhance(IMG_ARR)
    arr2image('pictures/enhanced.jpg', ANS_ARR)
    ANS_ARR = sharpen(IMG_ARR)
    arr2image('pictures/sharpened.jpg', ANS_ARR)
    ANS_ARR = edge(IMG_ARR)
    arr2image('pictures/edge.jpg', ANS_ARR)
    ANS_ARR = invert(edge(IMG_ARR))
    arr2image('pictures/sketch.jpg', ANS_ARR)
