import torch
import os
import glob
import shutil
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import json
import subprocess
import gc
import mediapipe
import pandas as pd
 
def resize_with_pad(im, target_width, target_height):
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    target_ratio = target_height / target_width
    im_ratio = im.height / im.width
    if target_ratio > im_ratio:
        # It must be fixed by width
        resize_width = target_width
        resize_height = round(resize_width * im_ratio)
    else:
        # Fixed by height
        resize_height = target_height
        resize_width = round(resize_height / im_ratio)

    image_resize = im.resize((resize_width, resize_height), Image.ANTIALIAS)
    background = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 255))
    offset = (round((target_width - resize_width) / 2), round((target_height - resize_height) / 2))
    background.paste(image_resize, offset)
    return background.convert('RGB')

    #Add pairs
def write_row(file_, *columns):
    print(*columns, sep='\t', end='\n', file=file_)


def resize_write_pad(im, target_width, target_height):
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    target_ratio = target_height / target_width
    im_ratio = im.height / im.width
    if target_ratio > im_ratio:
        # It must be fixed by width
        resize_width = target_width
        resize_height = round(resize_width * im_ratio)
    else:
        # Fixed by height
        resize_height = target_height
        resize_width = round(resize_height / im_ratio)

    image_resize = im.resize((resize_width, resize_height), Image.ANTIALIAS)
    background = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 255))
    offset = (round((target_width - resize_width) / 2), round((target_height - resize_height) / 2))
    background.paste(image_resize, offset)
    return background.convert('RGB')

def otsu(img , n  , x ):
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img_gray,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,n,x)
    return thresh

def contour(img):
    edges = cv2.dilate(cv2.Canny(img,200,255),None)
    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    mask = np.zeros((img.shape[0],img.shape[1]), np.uint8)
    masked = cv2.drawContours(mask, [cnt],-1, 255, -1)
    return masked

def get_cloth_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    return mask

def write_edge(C_path,E_path):
    img = cv2.imread(C_path)
    res = get_cloth_mask(img)
    if(np.mean(res)<100):
        ot = otsu(img,11,0.6)
        res = contour(ot)
    cv2.imwrite(E_path,res)

def main():
    files = glob.glob('input/*/*/*.*')
    for f in files:
        os.remove(f)

    files = glob.glob('results/*/*/*.*')
    for f in files:
        os.remove(f)

    upper = open('input/upper_body/test_pairs_unpaired.txt', 'w')
    lower = open('input/lower_body/test_pairs_unpaired.txt', 'w')
    dresses = open('input/dresses/test_pairs_unpaired.txt', 'w')
    all = open('input/test_pairs_paired.txt', 'w')

    # chmod +x /root/autodl-tmp/ladi/subprocess_1.py
    # chmod +x /root/autodl-tmp/ladi/preprocess/Self-Correction-Human-Parsing/simple_extractor.py
    # chmod +x /root/autodl-tmp/ladi/subprocess_2.py
    # chmod +x /root/autodl-tmp/ladi/src/inference.py
    # chmod +x /root/autodl-tmp/ladi/src/utils/val_metrics.py

    with open('images/test_pairs.txt', "r") as file:
        data = file.readlines()
        for line in data:
            word = line.split()
            org_path = 'images/humans/' + word[0]
            if(word[2] == '0'):
              write_row(upper,'0'+word[0],word[1])
              write_row(all,'0'+word[0],word[1],word[2])
              res_path = 'input/upper_body/images/0' + word[0]
            elif(word[2] == '1'):
              write_row(lower,'1'+word[0],word[1])
              write_row(all,'1'+word[0],word[1],word[2])
              res_path = 'input/lower_body/images/1' + word[0]
            elif(word[2] == '2'):
              write_row(dresses,'2'+word[0],word[1])
              write_row(all,'2'+word[0],word[1],word[2])
              res_path = 'input/dresses/images/2' + word[0]
            image = Image.open(org_path)
            new = resize_with_pad(image,384,512)
            new.save(res_path)

    upper.close()
    lower.close()
    dresses.close()
    all.close()


    command = "/root/autodl-tmp/ladi/subprocess_1.py"
    subprocess.call(command, shell=True)

    command = "/root/miniconda3/envs/ladi-vton/bin/python /root/autodl-tmp/ladi/preprocess/Self-Correction-Human-Parsing/simple_extractor.py --dataset 'atr' --model-restore '/root/autodl-tmp/ladi/preprocess/Self-Correction-Human-Parsing/checkpoints/final.pth' --input-dir '/root/autodl-tmp/ladi/input/upper_body/images/' --output-dir '/root/autodl-tmp/ladi/input/upper_body/label_maps/'"
    subprocess.call(command, shell=True)
    command = "/root/miniconda3/envs/ladi-vton/bin/python /root/autodl-tmp/ladi/preprocess/Self-Correction-Human-Parsing/simple_extractor.py --dataset 'atr' --model-restore '/root/autodl-tmp/ladi/preprocess/Self-Correction-Human-Parsing/checkpoints/final.pth' --input-dir '/root/autodl-tmp/ladi/input/lower_body/images/' --output-dir '/root/autodl-tmp/ladi/input/lower_body/label_maps/'"
    subprocess.call(command, shell=True)
    command = "/root/miniconda3/envs/ladi-vton/bin/python /root/autodl-tmp/ladi/preprocess/Self-Correction-Human-Parsing/simple_extractor.py --dataset 'atr' --model-restore '/root/autodl-tmp/ladi/preprocess/Self-Correction-Human-Parsing/checkpoints/final.pth' --input-dir '/root/autodl-tmp/ladi/input/dresses/images/' --output-dir '/root/autodl-tmp/ladi/input/dresses/label_maps/'"
    subprocess.call(command, shell=True)

    command = "/root/autodl-tmp/ladi/subprocess_2.py"
    subprocess.call(command, shell=True)

    pattern = 'input/*/dense/*'
    mp ={0: 0, 128: 18, 64: 4, 132: 19, 69: 5, 136: 20, 75: 6, 140: 21, 145: 22, 85: 9, 150: 23, 90: 10, 155: 24, 121: 16, 105: 13, 111: 14, 52: 2, 117: 15, 57: 3, 124: 17,
         2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 9: 9, 10: 10, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24}

    lut = np.zeros((256, 1), dtype=np.uint8)

    for i in range(0, 256):
        lut[i] = mp.get(i) or mp[min(mp.keys(), key = lambda key: abs(key-i))]

    for images in glob.glob(pattern):
        if images.endswith(".png"):
            image = cv2.imread(images, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(images, cv2.LUT(image, lut))

    files = glob.glob('input/*/*/*.*')
    for f in files:
        if f.endswith("_1.jpg") or f.endswith("_1.png"):
            os.remove(f)

    for c in ['dresses','upper_body','lower_body']:
      files = glob.glob('images/'+c+'/*.*')
      path = 'input/' + c + '/images/'
      for f in files:
        if f.endswith("_1.jpg"):
          res = path +os.path.basename(f)
          shutil.copy (f, res)
          image = Image.open(res)
          new = resize_with_pad(image,384,512)
          new.save(res)

    for s in ['upper_body','lower_body','dresses']:
      input_path = '/root/autodl-tmp/ladi/input/' + s + '/images/'
      output_path = '/root/autodl-tmp/ladi/input/'+ s + '/masks/'
      for images in glob.glob('*',root_dir = input_path):
          if images.endswith("_1.jpg"):
            write_edge(input_path + images , output_path+ os.path.splitext(images)[0] +".png")

    #test
    gc.collect()
    command = "/root/miniconda3/envs/ladi-vton/bin/python /root/autodl-tmp/ladi/src/inference.py --num_inference_steps 20 --dataset dresscode --dresscode_dataroot ./input  --output_dir ./results --test_order unpaired  --batch_size 3 --num_workers 2 --enable_xformers_memory_efficient_attention"
    subprocess.call(command, shell=True)
    
    #metrics
    # command = "/root/miniconda3/envs/ladi-vton/bin/python /root/autodl-tmp/ladi/src/utils/val_metrics.py --category lower_body --gen_folder ./results/unpaired --dataset dresscode --dresscode_dataroot ./input --test_order unpaired --batch_size 4 --workers 2"
    # subprocess.call(command, shell=True)

    # #results of DressCode
    # pattern = 'results/unpaired/*/*'
    # for images in glob.glob(pattern):
    #     if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
    #       cv2.imshow(cv2.imread(images, cv2.IMREAD_UNCHANGED))

########## Refinement ##

    dresscode = 'final'
    filepath = os.path.join('input', f"test_pairs_paired.txt")
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
    org_paths = sorted(
        [os.path.join('input',category, 'images', line.strip().split()[0]) for line in lines for category in['lower_body', 'upper_body', 'dresses'] if
         os.path.exists(os.path.join('input',category, 'images', line.strip().split()[0]))]
    )
    res_paths = sorted(
        [os.path.join('results/unpaired', category, name) for category in ['lower_body', 'upper_body', 'dresses'] for name in os.listdir(os.path.join('results/unpaired', category)) if
         os.path.exists(os.path.join('results/unpaired', category, name))]
    )
    
    assert len(org_paths) == len(res_paths)
    sz = len(org_paths)

    for iter in range(0, sz):
        org_img = cv2.imread(org_paths[iter])
        org_res = cv2.imread(res_paths[iter])
        h, w = int(org_img.shape[0]/2), org_img.shape[1]
        img = org_img[:h, :w]
        res = org_res[:h, :w]
        mp_face_mesh = mediapipe.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode = True)
        results = face_mesh.process(img[:, :, ::-1])
        if(results.multi_face_landmarks == None):
            print('miss')
            continue
        landmarks = results.multi_face_landmarks[0]
        df = pd.DataFrame(list(mp_face_mesh.FACEMESH_FACE_OVAL), columns = ['p1', 'p2'])
        routes_idx = []

        p1 = df.iloc[0]['p1']
        p2 = df.iloc[0]['p2']
        for i in range(0, df.shape[0]):
            obj = df[df['p1'] == p2]
            p1 = obj['p1'].values[0]
            p2 = obj['p2'].values[0]

            cur = []
            cur.append(p1)
            cur.append(p2)
            routes_idx.append(cur)

        routes = []
        for sid, tid in routes_idx:
            sxy = landmarks.landmark[sid]
            txy = landmarks.landmark[tid]

            source = (int(sxy.x * img.shape[1]), int(sxy.y * img.shape[0]))
            target = (int(txy.x * img.shape[1]), int(txy.y * img.shape[0]))

            routes.append(source)
            routes.append(target)

        mask = np.zeros((img.shape[0], img.shape[1]))
        mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
        mask = mask.astype(bool)
        res[mask] = img[mask]
        org_img[:h, :w] = img
        org_res[:h, :w] = res
        cv2.imwrite(res_paths[iter].replace('results/unpaired', 'final').replace('_0.jpg', '_' + dresscode + '.jpg'), org_res)  


if __name__ == '__main__':
    main()