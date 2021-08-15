import os
import csv
import sys
import argparse
import numpy as np
from PIL import Image
from itertools import islice

# List of folders for training, validation and test.
lv1_folder_name = {'Training' : 'Training',
                   'PublicTest' : 'PublicTest',
                   'PrivateTest' : 'PrivateTest'}

emotion_table = {0 : 'neutral', 
                 1 : 'happy', 
                 2 : 'surprised', 
                 3 : 'sad', 
                 4 : 'anger', 
                 5 : 'disgust', 
                 6 : 'fear', 
                 7 : 'contempt'}

def str_to_image(pixels):
    image_string = pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48, 48)
    return Image.fromarray(image_data)

def main(base_folder, fer_path, ferplus_path, mode):
    print("Start generating ferplus images.")
    
    for _, value in lv1_folder_name.items():
        lv1_folder_path = os.path.join(base_folder, value)
        if not os.path.exists(lv1_folder_path):
            os.makedirs(lv1_folder_path)
        for _, v in emotion_table.items():
            lv2_folder_path = os.path.join(lv1_folder_path, v)
            if not os.path.exists(lv2_folder_path):
                os.makedirs(lv2_folder_path)
    
    ferplus_entries = []
    with open(ferplus_path, 'r') as ferpluscsv:
        ferplus_rows = csv.reader(ferpluscsv, delimiter=',')
        for row in islice(ferplus_rows, 1, None):
            ferplus_entries.append(row)
 
    index = 0
    with open(fer_path,'r') as csvfile:
        fer_rows = csv.reader(csvfile, delimiter=',')
        for row in islice(fer_rows, 1, None):
            ferplus_row = ferplus_entries[index]
            file_name = ferplus_row[1].strip()
            file_emotion_raw = list(map(float, ferplus_row[2:len(ferplus_row)]))
            if ferplus_row[0] == 'Training':
                mode = mode
            else:
                mode = 'majority'
            file_emotion_lst = process_emo_data(file_emotion_raw, mode)
            idx = np.argmax(file_emotion_lst)
            if idx < len(emotion_table): # not unknown or non-face 
                file_emotion_lst = file_emotion_lst[:-2]
                file_emotion_lst = [int(float(i)/sum(file_emotion_lst)) for i in file_emotion_lst]
                file_emotion_tag = np.argmax(file_emotion_lst)
                if len(file_name) > 0:
                    image = str_to_image(row[1])
                    image_path = os.path.join(base_folder, lv1_folder_name[ferplus_row[0]], emotion_table[file_emotion_tag], file_name)
                    image.save(image_path, compress_level=0)                
            index += 1 
            
    print("Done...")

def process_emo_data(emotion_raw, mode):
    size = len(emotion_raw)
    emotion_unknown     = [0.0] * size
    emotion_unknown[-2] = 1.0

    # remove emotions with a single vote (outlier removal) 
    for i in range(size):
        if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
            emotion_raw[i] = 0.0

    sum_list = sum(emotion_raw)
    emotion = [0.0] * size 

    if mode == 'majority': 
        # find the peak value of the emo_raw list 
        maxval = max(emotion_raw) 
        if maxval > 0.5*sum_list: 
            emotion[np.argmax(emotion_raw)] = maxval 
        else: 
            emotion = emotion_unknown   # force setting as unknown 
    elif mode == 'probability' or mode == 'crossentropy':
        sum_part = 0
        count = 0
        valid_emotion = True
        while sum_part < 0.75*sum_list and count < 3 and valid_emotion:
            maxval = max(emotion_raw) 
            for i in range(size): 
                if emotion_raw[i] == maxval: 
                    emotion[i] = maxval
                    emotion_raw[i] = 0
                    sum_part += emotion[i]
                    count += 1
                    if i >= 8:  # unknown or non-face share same number of max votes 
                        valid_emotion = False
                        if sum(emotion) > maxval:   # there have been other emotions ahead of unknown or non-face
                            emotion[i] = 0
                            count -= 1
                        break
        if sum(emotion) <= 0.5*sum_list or count > 3: # less than 50% of the votes are integrated, or there are too many emotions, we'd better discard this example
            emotion = emotion_unknown   # force setting as unknown 

    return [float(i)/sum(emotion) for i in emotion]

if __name__ == "__main__":
    main("./datasets/fer2013plus", "./datasets/fer2013.csv", "./datasets/fer2013new.csv", "probability")