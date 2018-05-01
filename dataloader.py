import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cPickle as pickle

num_image = 1313
num_class = 15
img_size = 416.0
grid_size = 13

root_path = './data/train/'
class_path = []
img_path = []

for i in range(1, num_class+1):
    if i < 10:
        class_path.append(root_path + '0' + str(i) + '/')
        img_path.append(root_path + '0' + str(i) + '/rgb/')
    else:
        class_path.append(root_path + str(i) + '/')
        img_path.append(root_path + str(i) + '/rgb/')

def class_label_grid(class_index, img):
    """Load the 'S X S X number_of_classes' class label grid of image.

    Args:
    	class_index (int): index of a class (1-15)
        img: image array

    Returns:
        3D numpy array: S * S * num_class
    """
    grid = np.zeros((grid_size, grid_size, num_class))  
    step = int(img_size / grid_size)
    for x in range(grid_size):
        for y in range(grid_size):
            temp = img[y*step:(y+1)*step, x*step:(x+1)*step, :]
            if np.sum(temp) > 0:
                grid[y, x, class_index - 1] = 1    
    return grid


def image_loader(class_index):
    """Load all the images of one specific class.

    Args:
        class_index (int): index of a class (1-15)

    Returns:
        Image - 4D numpy array: 1313 * 416 * 416 * 3 
        Grid - 4D numpy array: 1313 * S * S * num_class
    """
    image = []
    grid = []
    for i in range(0, num_image):
    	img_name = img_path[class_index - 1]
        if i < 10:
            img_name = img_name + '000' + str(i) + '.png'
        elif i < 100:
            img_name = img_name + '00' + str(i) + '.png'
        elif i < 1000:
            img_name = img_name + '0' + str(i) + '.png'
        else:
            img_name = img_name + str(i) + '.png'

        # read image data
        # img = Image.open(img_name)  # image extension *.png,*.jpg
        # img = img.resize((int(img_size), int(img_size)), Image.ANTIALIAS)
        # img.save(img_name)
        img = mpimg.imread(img_name)
        # plt.imshow(img)
        # plt.show()
        # print(img.shape)
        # print(img[0,0,:])
        image.append(img)


        # generate grid 
        grid.append(class_label_grid(class_index, img))
    return image, grid

def ctrl_pt_loader(class_index):
    """Load all the control points of one specific class.

    Args:
        class_index (int): index of a class (1-15)

    Returns:
        2D numpy array: 1313 * 18 
    """
    fr = open(class_path[class_index - 1] + 'bb.pkl')
    total_bbx = pickle.load(fr)
    fr.close()

    # x = []
    # y = []
    crtl_pt = []
    for i in range(0, num_image):
        bbx = total_bbx[str(class_index)]
        # print (bbx)
        pt = [img_size/2, img_size/2]
        for j in range(len(bbx)):
            # temp_x, temp_y = coordinate_transform(bbx[j][0][0], bbx[j][1][0])
            temp_x = bbx[j][0][0]
            temp_y = bbx[j][1][0] 
            pt.append(temp_x)
            pt.append(temp_y)
        crtl_pt.append(pt)

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
    return crtl_pt

if __name__ == '__main__':
    image, grid = image_loader(1)
    pt = ctrl_pt_loader(1)
    print len(image)
    print image[0].shape
    print '----'
    print len(grid)
    print grid[0].shape
    print '----'
    print len(pt)
    print len(pt[0])


