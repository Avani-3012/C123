import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time

if (not os.environ.get('PYHTONHTTPSVERIFY','')and getattr(ssl,'_create_unverified_context',None)):
    ssl._create_default_https_context = ssl._create_unverified_context


X = np.load('image.npz')['arr_0']
Y = pd.read_csv('labels.csv') ['labels']
print(pd.Series(Y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
n_classes = len(classes)


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=9, train_size=7500, test_size=2500)

X_train_scaled = X_train/255
X_test_scaled = X_test/255

clf = LogisticRegression(solver='saga', multi_class='multinomial' ).fit(X_train_scaled,y_train)

y_pred = clf.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

cap = cv2.VideoCapture(0)

while(True):
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width = gray.shape
        upper_left = (int(width / 2 - 56), int(height / 2 - 56)) 
        bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
        cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)
        Roi = gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
        img_pil = Image.fromarray(Roi)
        img_bw = img_pil.convert('L')
        img_bw_resize = img_pil.resize((28,28),Image.ANTIALIAS)
        img_bw_resize_invert = PIL.ImageOps.invert(img_bw_resize)
        pixel_filter =20
        min_pixel = np.percentile(img_bw_resize,pixel_filter)
        img_bw_resize_invert_scale =np.clip(img_bw_resize_invert-min_pixel,0,255)
        max_pixel = np.max(img_bw_resize_invert)
        img_bw_resize_invert_scale = np.asarray(img_bw_resize_invert_scale)/max_pixel
        test_sample = np.array(img_bw_resize_invert_scale).reshape(1784)
        test_predict = clf.predict(test_sample)
        print(test_predict)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1)and 0xFF == ord('q'):
            break

    except Exception :
        pass
cv2.destroyAllWindows()
cap.release()   
