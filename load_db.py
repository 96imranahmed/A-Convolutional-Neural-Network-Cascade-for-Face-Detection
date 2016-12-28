import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from PIL import ImageDraw
import math
import time
from random import randint
import os
import shutil
from skimage.transform import pyramid_gaussian
from compiler.ast import flatten
from pycocotools.coco import COCO
import etc
import sys
import pickle

def load_db_test():
    
    print "Loading test db..."
    
    load_db_start = time.time()

    annot_dir = etc.db_dir + "FDDB-folds/"
    test_img = []
    test_annot = []
    test_face_num = []
       
    
    for f_num in xrange(1,10+1):
        
        print f_num, "/", 10, "folds is loading..."
        
        index = str(f_num).zfill(2)
        annot_file = annot_dir + "FDDB-fold-" + index + "-ellipseList.txt"
        
        fp = open(annot_file)
        raw_data = fp.readlines()
        
        stage = 0
        for parsed_data in raw_data:
                                
            if stage == 0:
                file_name = parsed_data.rstrip()
                stage = 1
            elif stage == 1:
                face_num = int(parsed_data)
                
                test_img.append(file_name)
                test_face_num.append(face_num)
                
                test_annot_line = [0 for r in xrange(face_num)]
                stage = 2

            elif stage == 2:
                splitted = parsed_data.split()
                
                y_half = max([float(splitted[0]) * math.cos(abs(float(splitted[2]))), float(splitted[0]) * math.sin(abs(float(splitted[2])))])
                x_half = max([float(splitted[1]) * math.sin(abs(float(splitted[2]))), float(splitted[1]) * math.cos(abs(float(splitted[2])))])
                
                left = float(splitted[3]) - x_half
                right = float(splitted[3]) + x_half
                upper = float(splitted[4]) - y_half
                lower = float(splitted[4]) + y_half
                

                test_annot_line[test_face_num[-1] - face_num] = [left, upper, right, lower]


                face_num -= 1
                if face_num == 0:
                    test_annot_line = [elem for elem in test_annot_line if type(elem) != int]
                    test_annot.append(test_annot_line)
                    stage = 0

        fp.close()
    
    print "Test: " + str(len(test_img))
    
    load_db_finish = time.time()
    print load_db_finish - load_db_start, "secs for loading db..."
    
    return [test_img, test_annot, test_face_num]

def dump_neg_imgs():
    print('Loading negatives....')
    neg_no_person_dir = etc.db_dir + '/neg/'
    for f in os.listdir(neg_no_person_dir):
        os.remove(neg_no_person_dir+f)
    neg_data_dir = etc.db_dir +'/train2014/'
    neg_ann_file = etc.db_dir +'/train2014_annotations/instances_train2014.json'
    cur_neg_limit = etc.db_neg_data_limit
    coco = COCO(neg_ann_file)
    sup_cat_excl_person = 'outdoor food indoor appliance sports animal vehicle furniture accessory electronic kitchen'
    sup_cat_nms = sup_cat_excl_person.split()
    excl_imgs = set(coco.getImgIds(catIds=coco.getCatIds(supNms=['person'])))
    cat_ids = coco.getCatIds(supNms=sup_cat_nms)
    img_id_dict = {}
    img_dict = {}
    for cat_id in cat_ids:
        img_id_dict[cat_id] = coco.getImgIds(catIds=[cat_id])
        img_dict[cat_id] = coco.loadImgs(img_id_dict[cat_id])
    img_added = set()
    coco = None
    neg_count = 0
    limit = True
    cur_i = 0
    while limit:
        check_looped = False
        for key in img_id_dict.keys():
            try_img = None
            if cur_i >= len(img_id_dict[key]): 
                continue
            cur_img_id = img_id_dict[key][cur_i]
            check_looped = True
            if not cur_img_id in excl_imgs:
                if not cur_img_id in img_added:
                    try_img = img_dict[key][cur_i]
                else: 
                    #print('Image already added, skipping')
                    continue
            else:
                #print('Image contains person, skipping') 
                continue
            if neg_count+1 > cur_neg_limit:
                limit = False
                break
            if len(try_img) == 0: 
                #print('Faulty image, skipping')
                continue
            try:
                img_added.add(cur_img_id)
                shutil.copy(neg_data_dir+try_img['file_name'], neg_no_person_dir)
                print('Processing negatives: ', neg_count, ' out of ', cur_neg_limit, cur_img_id, try_img['file_name'])
                neg_count+=1
            except:
                print('Error - skipping', sys.exc_info()[0])
                neg_count+=1
        cur_i += 1
        if check_looped == False:
            break

def full_load_db_head_train():
    print('Loading negatives....')
    neg_no_person_dir = etc.db_dir + '/neg/'
    neg_count = 0
    cur_neg_limit = etc.db_neg_data_limit
    neg_db = [0 for n in xrange(cur_neg_limit)]
    neg_img = [0 for n in xrange(cur_neg_limit)]
    limit = True
    cur_i = 0
    for filename in os.listdir(neg_no_person_dir):
        try:
            if neg_count+1>cur_neg_limit:
                break
            img = Image.open(neg_no_person_dir+filename) 
            if len(np.shape(np.asarray(img))) != 3:
                continue
            neg_img[neg_count] = img
            neg_db_line = np.zeros((etc.neg_per_img,etc.dim_12), np.float32)
            for neg_iter in xrange(etc.neg_per_img):
                                      
                rad_rand = randint(0,min(img.size[0],img.size[1])-1)
                while(rad_rand <= etc.face_minimum):
                    rad_rand = randint(0,min(img.size[0],img.size[1])-1)     

                x_rand = randint(0, img.size[0] - rad_rand - 1)
                y_rand = randint(0, img.size[1] - rad_rand - 1)

                neg_cropped_img = img.crop((x_rand, y_rand, x_rand + rad_rand, y_rand + rad_rand))
                neg_cropped_arr = etc.img2array(neg_cropped_img,etc.img_size_12)
                  
                neg_db_line[neg_iter,:] = neg_cropped_arr
                
            neg_db[neg_count] = neg_db_line
            neg_count+=1
            print('Processing negatives: ', neg_count, ' out of ', cur_neg_limit)
        except:
            print('Error - skipping!', sys.exc_info()[0])
            neg_count+=1

    neg_db = [elem for elem in neg_db if type(elem) != int]
    neg_db = np.vstack(neg_db)
    neg_img = [elem for elem in neg_img if type(elem) != int]

    print "Loading HollywoodHeads training db..."
    cur_limit = etc.db_data_limit
    annot_dir = etc.db_dir + "HollywoodHeads/Annotations/"
    img_dir = etc.db_dir + "HollywoodHeads/JPEGImages/"
    count = 0
    pos_db = [0 for _ in xrange(cur_limit*1)]
    for filename in os.listdir(annot_dir):
        if count < cur_limit:
            if filename.endswith("xml"):
                cur_file = os.path.join(annot_dir, filename)
                cur_size = []
                cur_objects = []
                image = None
                root = ET.parse(cur_file).getroot()
                img_name = root.find('filename').text
                size = root.find("size")
                for elem in size.iter():
                    cur_size.append(elem.text)
                del cur_size[0]
                if not cur_size[2] == '3':
                    #print('Invalid image, skipping')
                    continue   
                for obj in root.findall('object'):
                    bbox = obj.find('bndbox')
                    if bbox is None:
                        continue
                    cur = {}
                    for element in bbox.iter():
                        if element.tag == "bndbox":
                            pass
                        else:
                            cur[element.tag] = element.text
                    cur_objects.append(cur)
                if len(cur_objects) > 1:
                    #print('Skipping image, >1 heads')
                    continue
                if len(cur_objects)==0:
                    #print('Skipping image, no boundary box or error in format')
                    continue
                pos_db_line = np.zeros((2,etc.dim_12 + etc.dim_24 + etc.dim_48), np.float32)
                left = int(float(cur_objects[0]['xmin']))
                right = int(float(cur_objects[0]['xmax']))
                upper = int(float(cur_objects[0]['ymin']))
                lower = int(float(cur_objects[0]['ymax']))   
                img = Image.open(img_dir+img_name)   
                cropped_img = img.crop((left, upper, right, lower))
                flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT)
                cropped_arr_12 = etc.img2array(cropped_img,etc.img_size_12)
                flipped_arr_12 = etc.img2array(flipped_img,etc.img_size_12)
                cropped_arr_24 = etc.img2array(cropped_img,etc.img_size_24)
                flipped_arr_24 = etc.img2array(flipped_img,etc.img_size_24)
                cropped_arr_48 = etc.img2array(cropped_img,etc.img_size_48)
                flipped_arr_48 = etc.img2array(flipped_img,etc.img_size_48)
                #Create training image
                pos_db_line[0,:etc.dim_12] = cropped_arr_12
                pos_db_line[0,etc.dim_12:etc.dim_12+etc.dim_24] = cropped_arr_24
                pos_db_line[0,etc.dim_12+etc.dim_24:] = cropped_arr_48
                pos_db_line[1,:etc.dim_12] = flipped_arr_12
                pos_db_line[1,etc.dim_12:etc.dim_12+etc.dim_24] = flipped_arr_24
                pos_db_line[1,etc.dim_12+etc.dim_24:] = flipped_arr_48
                pos_db[count] = pos_db_line
                #print('Processing: ', img_name, cur_size, cur_objects)
                count+=1
                print('Processing training db: ', count-1, ' out of ', cur_limit)
        else:
            break
    pos_db = [elem for elem in pos_db if type(elem) != int]    
    pos_db = np.vstack(pos_db)
    print "Pos: " + str(np.shape(pos_db)[0])
    print "Neg: " + str(np.shape(neg_db)[0])
    
    return [pos_db, neg_db, neg_img]

def full_load_db_cali_head_train():
    print "Loading HollywoodHeads training db..."
    cur_limit = etc.db_data_limit
    annot_dir = etc.db_dir + "HollywoodHeads/Annotations/"
    img_dir = etc.db_dir + "HollywoodHeads/JPEGImages/"
    count = 0
    x_db = [0 for _ in xrange(cur_limit)]
    for filename in os.listdir(annot_dir):
        if count < cur_limit:
            if filename.endswith("xml"):
                cur_file = os.path.join(annot_dir, filename)
                cur_size = []
                cur_objects = []
                image = None
                root = ET.parse(cur_file).getroot()
                img_name = root.find('filename').text
                size = root.find("size")
                for elem in size.iter():
                    cur_size.append(elem.text)
                del cur_size[0]
                if not cur_size[2] == '3':
                    #print('Invalid image, skipping')
                    continue   
                for obj in root.findall('object'):
                    bbox = obj.find('bndbox')
                    if bbox is None:
                        continue
                    cur = {}
                    for element in bbox.iter():
                        if element.tag == "bndbox":
                            pass
                        else:
                            cur[element.tag] = element.text
                    cur_objects.append(cur)
                if len(cur_objects) > 1:
                    #print('Skipping image, >1 heads')
                    continue
                if len(cur_objects)==0:
                    #print('Skipping image, no boundary box or error in format')
                    continue
                left = int(float(cur_objects[0]['xmin']))
                right = int(float(cur_objects[0]['xmax']))
                upper = int(float(cur_objects[0]['ymin']))
                lower = int(float(cur_objects[0]['ymax'])) 
                img = Image.open(img_dir+img_name)
                # ImageDraw.Draw(img).rectangle((left, upper, right, lower), outline = "red")
                # img.show()
                if right >= img.size[0]:
                    right = img.size[0]-1
                if lower >= img.size[1]:
                    lower = img.size[1]-1
                x_db_list = [0 for _ in xrange(etc.cali_patt_num)]

                for si,s in enumerate(etc.cali_scale):
                    for xi,x in enumerate(etc.cali_off_x):
                        for yi,y in enumerate(etc.cali_off_y):
                            
                            new_left = left - x*float(right-left)/s
                            new_upper = upper - y*float(lower-upper)/s
                            new_right = new_left+float(right-left)/s
                            new_lower = new_upper+float(lower-upper)/s
                            
                            new_left = int(new_left)
                            new_upper = int(new_upper)
                            new_right = int(new_right)
                            new_lower = int(new_lower)


                            if new_left < 0 or new_upper < 0 or new_right >= img.size[0] or new_lower >= img.size[1]:
                                continue

                            cropped_img = img.crop((new_left, new_upper, new_right, new_lower))
                            calib_idx = si*len(etc.cali_off_x)*len(etc.cali_off_y)+xi*len(etc.cali_off_y)+yi

                            #for debugging
                            #cropped_img.save(etc.pos_dir + str(i)  + ".jpg")

                            x_db_list[calib_idx] = [cropped_img,calib_idx]

                    
                x_db_list = [elem for elem in x_db_list if type(elem) != int]
                if len(x_db_list) > 0:
                    x_db[count] = x_db_list
                count+=1
                print('Processing training db: ', count-1, ' out of ', cur_limit)
        else:
            break

    x_db = [elem for elem in x_db if type(elem) != int]    
    x_db = [x_db[i][j] for i in xrange(len(x_db)) for j in xrange(len(x_db[i]))]
    return x_db
     
def load_db_detect_train():
   
    print "Loading training db..."

    annot_dir = etc.db_dir + "AFLW/aflw/data/"
    annot_fp = open(annot_dir + "annot", "r")
    raw_data = annot_fp.readlines()

    #pos image cropping
    pos_db = [0 for _ in xrange(len(raw_data))]
    for i,line in enumerate(raw_data):
        print i, "/", len(raw_data), "th pos image cropping..."
        parsed_line = line.split(',')
        img = Image.open(etc.pos_dir+parsed_line[0])
        #check if not RGB
        if len(np.shape(np.asarray(img))) != 3 or np.shape(np.asarray(img))[2] != etc.input_channel:
            continue
        pos_db_line = np.zeros((2,etc.dim_12 + etc.dim_24 + etc.dim_48), np.float32)
        left = int(parsed_line[1])
        upper = int(parsed_line[2])
        right = int(parsed_line[3]) + left
        lower = int(parsed_line[4]) + upper
       
        if right >= img.size[0]:
            right = img.size[0]-1
        if lower >= img.size[1]:
            lower = img.size[1]-1
        
        cropped_img = img.crop((left, upper, right, lower))
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT)

        cropped_arr_12 = etc.img2array(cropped_img,etc.img_size_12)
        flipped_arr_12 = etc.img2array(flipped_img,etc.img_size_12)
        cropped_arr_24 = etc.img2array(cropped_img,etc.img_size_24)
        flipped_arr_24 = etc.img2array(flipped_img,etc.img_size_24)
        cropped_arr_48 = etc.img2array(cropped_img,etc.img_size_48)
        flipped_arr_48 = etc.img2array(flipped_img,etc.img_size_48)
        
        #for debugging
        #cropped_img.save(etc.pos_dir + str(i)  + ".jpg")
        
        pos_db_line[0,:etc.dim_12] = cropped_arr_12
        pos_db_line[0,etc.dim_12:etc.dim_12+etc.dim_24] = cropped_arr_24
        pos_db_line[0,etc.dim_12+etc.dim_24:] = cropped_arr_48
        
        
        pos_db_line[1,:etc.dim_12] = flipped_arr_12
        pos_db_line[1,etc.dim_12:etc.dim_12+etc.dim_24] = flipped_arr_24
        pos_db_line[1,etc.dim_12+etc.dim_24:] = flipped_arr_48
        
        pos_db[i] = pos_db_line

    pos_db = [elem for elem in pos_db if type(elem) != int]    
    pos_db = np.vstack(pos_db) 

    #neg image cropping
    nid = 0
    neg_file_list = os.listdir(etc.neg_dir)
    neg_db = [0 for n in xrange(len(neg_file_list))]
    neg_img = [0 for n in xrange(len(neg_file_list))]
    for file in neg_file_list:
        img = Image.open(etc.neg_dir + file)
        if len(np.shape(np.asarray(img))) != 3:
            continue
        neg_img[nid] = img
        print nid+1, "/" , len(neg_file_list), "th neg image cropping..."
        neg_db_line = np.zeros((etc.neg_per_img,etc.dim_12), np.float32)
        for neg_iter in xrange(etc.neg_per_img):
                     
            rad_rand = randint(0,min(img.size[0],img.size[1])-1)
            while(rad_rand <= etc.face_minimum):
                rad_rand = randint(0,min(img.size[0],img.size[1])-1)
            
            x_rand = randint(0, img.size[0] - rad_rand - 1)
            y_rand = randint(0, img.size[1] - rad_rand - 1)
            
            neg_cropped_img = img.crop((x_rand, y_rand, x_rand + rad_rand, y_rand + rad_rand))
            neg_cropped_arr = etc.img2array(neg_cropped_img,etc.img_size_12)
            
            #for debugging
            #neg_cropped_img.save(etc.neg_dir + str(f_num) + "_" + str(r) + ".jpg")
                   
            neg_db_line[neg_iter,:] = neg_cropped_arr
        
        neg_db[nid] = neg_db_line
        nid += 1
   
    neg_db = [elem for elem in neg_db if type(elem) != int]
    neg_db = np.vstack(neg_db)
    neg_img = [elem for elem in neg_img if type(elem) != int]
     
    print "Pos: " + str(np.shape(pos_db)[0])
    print "Neg: " + str(np.shape(neg_db)[0])
    
    return [pos_db, neg_db, neg_img]


def load_db_cali_train():
   
    print "Loading training db..."

    annot_dir = etc.db_dir + "AFLW/aflw/data/"
    annot_fp = open(annot_dir + "annot", "r")
    raw_data = annot_fp.readlines()
    
    #pos image cropping
    x_db = [0 for _ in xrange(len(raw_data))]
    for i,line in enumerate(raw_data):
        
        print i, "/", len(raw_data), "th pos image cropping..."
        parsed_line = line.split(',')
        img = Image.open(etc.pos_dir+parsed_line[0])
        
        #check if not RGB
        if len(np.shape(np.asarray(img))) != 3 or np.shape(np.asarray(img))[2] != etc.input_channel:
            continue
        
        left = int(parsed_line[1])
        upper = int(parsed_line[2])
        right = int(parsed_line[3]) + left
        lower = int(parsed_line[4]) + upper
       
        if right >= img.size[0]:
            right = img.size[0]-1
        if lower >= img.size[1]:
            lower = img.size[1]-1
        
        x_db_list = [0 for _ in xrange(etc.cali_patt_num)]

        for si,s in enumerate(etc.cali_scale):
            for xi,x in enumerate(etc.cali_off_x):
                for yi,y in enumerate(etc.cali_off_y):
                    
                    new_left = left - x*float(right-left)/s
                    new_upper = upper - y*float(lower-upper)/s
                    new_right = new_left+float(right-left)/s
                    new_lower = new_upper+float(lower-upper)/s
                    
                    new_left = int(new_left)
                    new_upper = int(new_upper)
                    new_right = int(new_right)
                    new_lower = int(new_lower)


                    if new_left < 0 or new_upper < 0 or new_right >= img.size[0] or new_lower >= img.size[1]:
                        continue

                    cropped_img = img.crop((new_left, new_upper, new_right, new_lower))
                    calib_idx = si*len(etc.cali_off_x)*len(etc.cali_off_y)+xi*len(etc.cali_off_y)+yi

                    #for debugging
                    #cropped_img.save(etc.pos_dir + str(i)  + ".jpg")

                    x_db_list[calib_idx] = [cropped_img,calib_idx]

            
        x_db_list = [elem for elem in x_db_list if type(elem) != int]
        if len(x_db_list) > 0:
            x_db[i] = x_db_list

    x_db = [elem for elem in x_db if type(elem) != int]    
    x_db = [x_db[i][j] for i in xrange(len(x_db)) for j in xrange(len(x_db[i]))]
    
    return x_db

def proc_positive_list(limit = 100000):
    count = 0
    annot_dir = etc.db_dir + "HollywoodHeads/Annotations/"
    img_dir = etc.db_dir + "HollywoodHeads/JPEGImages/"
    data_db = [0 for _ in xrange(limit)]
    if os._exists(etc.db_dir + 'data_db.pickle'): os.remove(etc.db_dir + 'data_db.pickle')
    for filename in os.listdir(annot_dir):
        if count < limit:
            if filename.endswith("xml"):
                cur_file = os.path.join(annot_dir, filename)
                cur_size = []
                cur_objects = []
                image = None
                root = ET.parse(cur_file).getroot()
                img_name = root.find('filename').text
                size = root.find("size")
                for elem in size.iter():
                    cur_size.append(elem.text)
                del cur_size[0]
                if not cur_size[2] == '3':
                    #print('Invalid image, skipping')
                    continue   
                for obj in root.findall('object'):
                    bbox = obj.find('bndbox')
                    if bbox is None:
                        continue
                    cur = {}
                    for element in bbox.iter():
                        if element.tag == "bndbox":
                            pass
                        else:
                            cur[element.tag] = element.text
                    cur_objects.append(cur)
                if len(cur_objects) > 1:
                    #print('Skipping image, >1 heads')
                    continue
                if len(cur_objects)==0:
                    #print('Skipping image, no boundary box or error in format')
                    continue
                left = int(float(cur_objects[0]['xmin']))
                right = int(float(cur_objects[0]['xmax']))
                upper = int(float(cur_objects[0]['ymin']))
                lower = int(float(cur_objects[0]['ymax']))
                data_db[count] = tuple([img_name, left, upper, right, lower])
                print("Processing: ", count)
                count+=1
    data_db = [elem for elem in data_db if type(elem) != int]
    with open(etc.db_dir + 'data_db.pickle', 'wb') as handle:
        pickle.dump(data_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Proccessed array dumped to file')

def load_positive_list():
    with open(etc.db_dir + 'data_db.pickle', 'rb') as handle:
        return pickle.load(handle)

def main(argv):
    #dump_neg_imgs()
    # proc_positive_list(limit = 150000)
    # print(load_positive_list())
    print('Loading databases')

if __name__ == "__main__":
    main(sys.argv)