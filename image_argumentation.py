from math import pi, log10
import zipfile
import numpy as np
import os
import sys
import cv2
import time
import argparse

#Full Code Demo
'''
def transformImage(img):
    transformer = ImageTransformer()
    save_image(imgPath, "GaussianBlur", cv2.GaussianBlur(img,(5,5), 0))
    save_image(imgPath, "Salt_Pepper_noise", transformer.addnoise(img, 0.15))

    for ang in range(-80, 80, 10):
        save_image(imgPath, "Rotate_" + str(ang), transformer.rotate_along_axis(img, phi = ang))

    for temp in range(10, 100, 5):
        gamma = temp / 100.0
        save_image(imgPath, "add_light", transformer.add_light(img, gamma))
        save_image(imgPath, "add_light_color moss green", transformer.add_light_color(img, 68, gamma))
        save_image(imgPath, "add_light_color sea blue", transformer.add_light_color(img, 197, gamma))

    for temp in range(5, 50, 5):
        gamma = temp / 10.0
        save_image(imgPath, "add_light", transformer.add_light(img, gamma))

        save_image(imgPath, "add_light_color moss green", transformer.add_light_color(img, 75, gamma))
        save_image(imgPath, "add_light_color sea blue", transformer.add_light_color(img, 197, gamma))

        for colour in range(100, 250, 50):
            save_image(imgPath, "add_light_color", transformer.add_light_color(img, colour, gamma))

    for saturation in range(50, 200, 50):
        save_image(imgPath, "saturation_image", transformer.saturation_image(img, saturation))

    for hue in range(50, 200, 10):
        save_image(imgPath, "hue_image", transformer.hue_image(img, hue))

    save_image(imgPath, "custom_scale", transformer.scale_image(img,0.3,0.3))
    save_image(imgPath, "scale", transformer.scale_image(img,2,2))

    save_image(imgPath, "rotate", transformer.rotate_image(img,90))
'''

def main():
    args = get_args()
    successful_count = 0
    transformer = ImageTransformer()
    list_of_files = getFiles(args.image_source, args.image_format)

    img_target_folder = os.getcwd() + '/' + args.target_folder + args.image_source
    xml_target_folder = os.getcwd() + '/' + args.target_folder + args.annotation_source

    for idx, imgPath in enumerate(list_of_files):
        imgName = os.path.basename(imgPath)
        progress(idx, len(list_of_files), suffix='Reading: {}'.format(imgName))

        xmlfilepath = findFile(os.path.splitext(imgName)[0] + '.xml', args.annotation_source)
        if xmlfilepath is None:
            continue

        progress(idx, len(list_of_files), suffix='Editing: {}'.format(imgName))
        successful_count += 1

        img = read_image(imgPath)
        xmlfile = read_xml(xmlfilepath)

        save_data("blur", cv2.GaussianBlur(img,(31,31), 0),    xmlfile, imgName, img_target_folder, xml_target_folder)
        save_data("noise", transformer.addnoise(img, 0.20),    xmlfile, imgName, img_target_folder, xml_target_folder)
        save_data("light18", transformer.add_light(img, 1.8),  xmlfile, imgName, img_target_folder, xml_target_folder)
        save_data("light5", transformer.add_light(img, 0.5),   xmlfile, imgName, img_target_folder, xml_target_folder)

    progress(len(list_of_files), len(list_of_files), suffix='DONE: Argumented {} images\n'.format(successful_count))

    if args.auto_zip:
        for funcnamelist in ["noise", ["blur", "light18", "light5"]]:

            if type(funcnamelist) == str:
                funcnamelist = [funcnamelist]

            zipfilename = 'argumented_images_' + '_'.join(funcnamelist) + '.zip'
            with zipfile.ZipFile(zipfilename, 'w', zipfile.ZIP_DEFLATED, 9) as zip:
                img_files = []
                xml_files = []

                for funcname in funcnamelist:
                    img_files.extend(getFiles(args.target_folder + args.image_source, '_' + funcname + args.image_format))
                    xml_files.extend(getFiles(args.target_folder + args.annotation_source, '_' + funcname + '.xml'))

                # writing each file one by one
                count = 0
                totalcount = len(img_files) + len(xml_files)

                for file in img_files:
                    count += 1
                    filename = os.path.basename(file)
                    progress(count, totalcount, suffix='Archiving: {}'.format(filename))
                    zip.write(file, arcname=(args.image_source + filename))

                for file in xml_files:
                    count += 1
                    filename = os.path.basename(file)
                    progress(count, totalcount, suffix='Archiving: {}'.format(filename))
                    zip.write(file, arcname=(args.annotation_source + filename))

            progress(totalcount, totalcount, suffix='DONE: Archived \'{}\' argumented images\n'.format('_'.join(funcnamelist)))
    print('[========== COMPLETED ==========]')

def get_args():
    parser = argparse.ArgumentParser(description='A Simple Image Argumentation Tool by ShawnCX')
    parser.add_argument('-z', '--zip',          dest='auto_zip',          help='Automatically zip files by argumentation type', action='store_true')
    parser.add_argument('-i', '--image',        dest='image_source',      help='Specify path to image source folder',           action='store', default='images/')
    parser.add_argument('-a', '--annotation',   dest='annotation_source', help='Specify path to annotation source folder',      action='store', default='annotations/')
    parser.add_argument("-t", '--target',       dest='target_folder',     help='Specify path to target folder',                 action='store', default='output/')
    parser.add_argument("-f", '--image_format', dest='image_format',      help='Specify image format',                          action='store', default='.jpg')
    return parser.parse_args()

def progress(count, total, suffix=''):
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = str(round(100.0 * count / float(total), 1)) + '%'
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    #CURSOR_UP_ONE = '\x1b[1A'
    #ERASE_LINE = '\x1b[2K'
    count = str(count).rjust(int(log10(total) + 1))

    sys.stdout.write('\r\x1b[2K')
    sys.stdout.write('[%s of %s] [%s] %s %s' % (count, total, bar, percents.ljust(6), suffix))
    sys.stdout.flush()


""" IO Functions """
def read_image(img_path, abs_bbox = None):
    img = cv2.imread(img_path)

    if abs_bbox is not None:
        img = img[abs_bbox[0][1]:abs_bbox[1][1], abs_bbox[0][0]:abs_bbox[1][0]]

    return img

def read_xml(xmlpath):
    return open(xmlpath).read()

def save_data(func_name, img, xmlfile, filename, img_target_folder, xml_target_folder):
    extension = os.path.splitext(filename)[1]
    newimgfilename = filename.replace(extension, '_' + func_name + extension)

    imgpath = img_target_folder + newimgfilename
    xmlpath = xml_target_folder + filename.replace(extension, '_' + func_name + '.xml')

    if not os.path.exists(img_target_folder):
        os.makedirs(img_target_folder)
    if not os.path.exists(xml_target_folder):
        os.makedirs(xml_target_folder)

    #cv2.imshow("TEST", img)
    #cv2.waitKey(400)
    cv2.imwrite(imgpath, img)

    f = open(xmlpath, 'w')
    f.write(xmlfile.replace(filename, newimgfilename))
    f.close()

def getFiles(directory='', fileType='.jpg', startWithName=''):
    res = []
    directory = os.getcwd() + '/' + directory
    for currentfile in os.listdir(directory):
        if currentfile.lower().startswith(startWithName.lower()) and currentfile.endswith(fileType):
            res.append(os.path.join(directory, currentfile))

    return sorted(res)

def findFile(name, directory=''):
    target_file = os.getcwd() + '/' + directory + name
    return target_file if os.path.exists(target_file) else None


""" Image Utility Functions """
def get_rad(theta, phi, gamma):
    return (deg_to_rad(theta),
            deg_to_rad(phi),
            deg_to_rad(gamma))

def get_deg(rtheta, rphi, rgamma):
    return (rad_to_deg(rtheta),
            rad_to_deg(rphi),
            rad_to_deg(rgamma))

def deg_to_rad(deg):
    return deg * pi / 180.0

def rad_to_deg(rad):
    return deg * 180.0 / pi

# Parameters:
#     image_path: the path of image that you want rotated
#     shape     : the ideal shape of input image, None for original size.
#     theta     : rotation around the x axis
#     phi       : rotation around the y axis
#     gamma     : rotation around the z axis (basically a 2D rotation)
#     dx        : translation along the x axis
#     dy        : translation along the y axis
#     dz        : translation along the z axis (distance to the image)
#
# Output:
#     image     : the rotated image

""" Perspective transformation class for image"""
class ImageTransformer(object):

    """ Wrapper of Rotating a Image """
    def rotate_along_axis(self, img, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):
        
        # Get radius of rotation along 3 axes
        rtheta, rphi, rgamma = get_rad(theta, phi, gamma)
        
        # Get ideal focal length on z axis
        # NOTE: Change this section to other axis if needed
        height, width = img.shape[0:2]
        d = np.sqrt(height**2 + width**2)
        focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)

        # Get projection matrix
        mat = self.get_M(width, height, focal, rtheta, rphi, rgamma, dx, dy, focal)
        image = cv2.warpPerspective(img.copy(), mat, (width, height), borderMode=cv2.BORDER_TRANSPARENT)
        return image

    def get_M(self, width, height, focal, theta, phi, gamma, dx, dy, dz):
        
        w = width
        h = height
        f = focal

        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, 1]])
        
        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([ [1, 0, 0, 0],
                        [0, np.cos(theta), -np.sin(theta), 0],
                        [0, np.sin(theta), np.cos(theta), 0],
                        [0, 0, 0, 1]])
        
        RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                        [0, 1, 0, 0],
                        [np.sin(phi), 0, np.cos(phi), 0],
                        [0, 0, 0, 1]])
        
        RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                        [np.sin(gamma), np.cos(gamma), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([  [1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([ [f, 0, w/2, 0],
                        [0, f, h/2, 0],
                        [0, 0, 1, 0]])

        # Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(R, A1)))

    @staticmethod
    def brighten(image,value):
        return np.where((255 - image) < value,255,image+value)
        
    @staticmethod
    def darken(image,value):
        return np.where(image < value,0,image-value)

    @staticmethod
    def scale_image(image,fx,fy):
        image = cv2.resize(image,None,fx=fx, fy=fy, interpolation = cv2.INTER_CUBIC)
        return image

    @staticmethod
    def translation_image(image,x,y):
        rows, cols ,c= image.shape
        M = np.float32([[1, 0, x], [0, 1, y]])
        image = cv2.warpAffine(image, M, (cols, rows))
        return image

    @staticmethod
    def rotate_image(image,deg):
        rows, cols,c = image.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2), deg, 1)
        image = cv2.warpAffine(image, M, (cols, rows))
        return image

    @staticmethod
    def invert_image(image,channel=255):
        # image=cv2.bitwise_not(image)
        return (channel-image)
    
    @staticmethod
    def add_light(image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                        for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(image, table)

    @staticmethod
    def add_light_color(image, color, gamma=1.0):
        invGamma = 1.0 / gamma
        image = (color - image)
        table = np.array([((i / 255.0) ** invGamma) * 255
                        for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(image, table)

    @staticmethod
    def saturation_image(image,saturation):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        v = image[:, :, 2]
        v = np.where(v <= 255 - saturation, v + saturation, 255)
        image[:, :, 2] = v

        return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    @staticmethod
    def hue_image(image,saturation):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        v = image[:, :, 2]
        v = np.where(v <= 255 + saturation, v - saturation, 255)
        image[:, :, 2] = v

        return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    @staticmethod
    def addnoise(image, ratio=0.15):
        row,col,ch = image.shape
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(image.size * ratio)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                 for i in image.shape]
        out[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(image.size * ratio)
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[tuple(coords)] = 0
        return out

if __name__ == "__main__":
    main()
