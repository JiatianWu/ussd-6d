import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cPickle as pickle

# import cv2


num_class = 15
num_image = 1313
img_size = 416.0
grid_size = 13

w_ori = 480.0
h_ori = 640.0

root_path = './train/'
class_path = []
img_path = []


def coordinate_transform(x, y):
    # new_x = x * 1.0 / w_ori * img_size
    # new_y = y * 1.0 / h_ori * img_size
    new_x = x * 1.0 / h_ori * img_size
    new_y = y * 1.0 / w_ori * img_size
    return new_x, new_y


for i in range(1, num_class+1):
    if i < 10:
        class_path.append(root_path + '0' + str(i) + '/')
        img_path.append(root_path + '0' + str(i) + '/rgb/')
    else:
        class_path.append(root_path + str(i) + '/')
        img_path.append(root_path + str(i) + '/rgb/')

# print (class_path)


for idx_class in range(1, num_class+1):
    # import all the bbx of class idx_class in dictionary format
    fr = open(class_path[idx_class - 1] + 'bb.pkl')
    total_bbx = pickle.load(fr)
    fr.close()

    data = []
    # genarate image name
    for i in range(0,num_image):
        img_name = img_path[idx_class - 1]
        if i < 10:
            img_name = img_name + '000' + str(i) + '.png'
        elif i < 100:
            img_name = img_name + '00' + str(i) + '.png'
        elif i < 1000:
            img_name = img_name + '0' + str(i) + '.png'
        else:
            img_name = img_name + str(i) + '.png'

        # read image data
        img = Image.open(img_name)  # image extension *.png,*.jpg
        img = img.resize((int(img_size), int(img_size)), Image.ANTIALIAS)
        img.save(img_name)
        img = mpimg.imread(img_name)
        # plt.imshow(img)
        # plt.show()
        # print(img.shape)
        # print(img[0,0,:])

        # form 9 control points
        bbx = total_bbx[str(i)]
        pt = [img_size/2, img_size/2]
        # print (bbx)

        # x = []
        # y = []
        for j in range(len(bbx)):
            temp_x, temp_y = coordinate_transform(bbx[j][0][0], bbx[j][1][0])
            pt.append(temp_x)
            pt.append(temp_y)
            # pt.append(bbx[j][0][0] * 1.0 / h_ori * img_size)
            # pt.append(bbx[j][1][0] * 1.0 / w_ori * img_size)
            # x.append(temp_x)
            # y.append(temp_y)
            
        # print (pt)
        # test 3D bbx
        # plt.scatter(x,y)
        # plt.plot([x[0],x[1]], [y[0],y[1]], '-o')
        # plt.plot([x[0],x[2]], [y[0],y[2]], '-o')
        # plt.plot([x[0],x[4]], [y[0],y[4]], '-o')
        # plt.plot([x[1],x[3]], [y[1],y[3]], '-o')
        # plt.plot([x[1],x[5]], [y[1],y[5]], '-o')
        # plt.plot([x[2],x[3]], [y[2],y[3]], '-o')
        # plt.plot([x[2],x[6]], [y[2],y[6]], '-o')
        # plt.plot([x[3],x[7]], [y[3],y[7]], '-o')
        # plt.plot([x[4],x[5]], [y[4],y[5]], '-o')
        # plt.plot([x[4],x[6]], [y[4],y[6]], '-o')
        # plt.plot([x[5],x[7]], [y[5],y[7]], '-o')
        # plt.plot([x[6],x[7]], [y[6],y[7]], '-o')
        # plt.show()

        # form class in each grid
        grid = np.zeros((grid_size, grid_size))  
        step = int(img_size / grid_size)
        for x in range(grid_size):
            for y in range(grid_size):
                temp = img[y*step:(y+1)*step, x*step:(x+1)*step, :]
                if np.sum(temp) > 0:
                    grid[y][x] = idx_class
        # print grid

        # form a datapoint 
        datapoint = {}
        # datapoint['image'] = img
        datapoint['coordinates'] = pt
        datapoint['grid'] = grid
        data.append(datapoint)
        # print datapoint
    print (idx_class)
    np.save('./' + str(idx_class) + '.npy', data)