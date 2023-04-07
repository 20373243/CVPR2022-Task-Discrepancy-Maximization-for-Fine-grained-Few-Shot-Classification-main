import os
from PIL import Image


## 遍历一个文件夹下的所有图像 
def bianli_pics(path):
    img_folder = path
    #print('img_folder:  ',img_folder)
    for indexp,people in enumerate(os.listdir(img_folder)):
        #print('people: ',people)
        people_path = os.path.join(img_folder, people)
        print('peoplepath: ',people_path)
        for default in os.listdir(people_path):
                #print('default: ',default)
                default_path = os.path.join(people_path,default)
                #print('defaultpath: ',default_path)
                img_list = [os.path.join(name) for name in os.listdir(default_path) if name[-3:] in ['jpg', 'png', 'gif']]
                length = len(img_list) * 0.65
                for index, i in enumerate(img_list):
                    if index > length:
                        break
                    path_new=os.path.join(default_path,i)
                    target_path = os.path.join('C:\\Dataset/medical/1_new/',str(indexp)+str(i))
                    image = Image.open(path_new)
                    image = image.crop((170, 170, 170+150, 170+150))
                    image.save(target_path)
            

  


if __name__=="__main__":
	path="C:\\Dataset/medical/1/"
	bianli_pics(path)
