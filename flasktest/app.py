from flask import Flask, redirect, render_template, url_for

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/audio")
def audio():
    return "audio processing"

@app.route("/image")
def image():
    return "image processing"