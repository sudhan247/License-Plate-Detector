import os
import re
import requests
from bs4 import BeautifulSoup, SoupStrainer
import re
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import io
import json
from PIL import Image
from PIL import ImageFilter
from collections  import *
garbage=[]
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def cannyget(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
def mod(arg):
    global garbage
    ind=[[] for _ in range(len(arg))]
    garbage=[[] for _ in range(len(arg))]
    for i in range(5):
        for j in range(len(arg)):
            image=Image.open(arg[j])
            image.filter(ImageFilter.SHARPEN)
            result = pytesseract.image_to_string(image,lang='eng', config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            result=''.join(result.split())
            date_pattern = r'[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}'
            ind[j].extend(re.findall(date_pattern, result))
            garbage[j].append(result)
    mod_plates=[]
    for j in range(len(arg)):
        if len(ind[j]):
            mod_plates.append(ind[j][0])
        else:
            mod_plates.append(' '*10)
    return mod_plates
    
home_url = 'https://parivahan.gov.in/rcdlstatus/?pur_cd=102'
post_url = 'https://parivahan.gov.in/rcdlstatus/vahan/rcDlHome.xhtml'
def find(plates):
    files=[i for i in plates]
    plates=mod(plates)
    found=0
    for plate in range(len(plates)):
        if plates[plate]!=' '*10:
            first = plates[plate][:-4]
            second = plates[plate][-4:]
            r = requests.get(url=home_url)
            cookies = r.cookies
            soup = BeautifulSoup(r.text, 'html.parser')
            viewstate = soup.select('input[name="javax.faces.ViewState"]')[0]['value']
            i = 0
            for match in soup.find_all('button', id=re.compile("form_rcdl")):
                if i ==  0:
                    button_id= match.get('id')
                i = 1

            data = {
                  'javax.faces.partial.ajax':'true',
                  'javax.faces.source':button_id,
                  'javax.faces.partial.execute':'@all',
                  'javax.faces.partial.render': 'form_rcdl:pnl_show form_rcdl:pg_show form_rcdl:rcdl_pnl',
                  button_id:button_id,
                  'form_rcdl':'form_rcdl',
                  'form_rcdl:tf_reg_no1': first,
                  'form_rcdl:tf_reg_no2': second,
                  'javax.faces.ViewState': viewstate,
              }

            r = requests.post(url=post_url, data=data, cookies=cookies)
            soup = BeautifulSoup(r.text, 'html.parser')
            table = SoupStrainer('tr')
            soup = BeautifulSoup(soup.get_text(), 'html.parser', parse_only=table)
            print('image {0}/{1} {2}'.format(plate+1,len(files),files[plate]))
            print(soup.get_text())
            found+=1
        else:
            print('image {0}/{1} {2}'.format(plate+1,len(files),files[plate]))
            print('Character Recognition Failed')
            print('Wasted Results:',garbage[plate])
        print('*'*50)
    print('{0}/{1} Passed'.format(found,len(files)))

