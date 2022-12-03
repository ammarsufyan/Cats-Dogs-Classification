import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
from tensorflow.keras.utils import load_img, img_to_array
import pandas as pd
from tqdm import tqdm
import os
from keras.models import load_model

model = load_model('model_catsVSdogs_10epoch.h5')

# initialise GUI
top = tk.Tk()
top.geometry('800x600')
top.title('CatsVSDogs Classification')
top.configure(background='#CDCDCD')
label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)


def classify(file_path):
    global label_packed
    result = "Unknown"
    # predicting images
    img = load_img(file_path, target_size=(150, 150))
    x = img_to_array(img)
    x /= 255
    x = numpy.expand_dims(x, axis=0)
    images = numpy.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes)
    if classes[0] > 0.5:
        result = "Dogs"
    else:
        result = "Cats"

    label.configure(foreground='#011638', text=result)

# classify the image using model .h5


def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image",
                        command=lambda: classify(file_path),
                        padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white',
                         font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),
                            (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload = Button(top, text="Upload an image",
                command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white',
                 font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="CatsVSDogs Classification",
                pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()
