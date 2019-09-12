import os
import time

def main():

    img_delete_count = 0
    xml_delete_count = 0

    for imgPath in getFiles('images/', '.jpg'):
        imgName = os.path.basename(imgPath)

        xmlfile = findFile((os.path.splitext(imgName)[0] + '.xml'), 'annotations/')
        if xmlfile is None:
            print('Delete: {}'.format(imgName))
            os.remove(imgPath)
            img_delete_count += 1

    for xmlPath in getFiles('annotations/', '.xml'):
        xmlName = os.path.basename(xmlPath)

        imgfile = findFile((os.path.splitext(xmlName)[0] + '.jpg'), 'images/')
        if imgfile is None:
            print('Delete: {}'.format(xmlName))
            os.remove(xmlPath)
            xml_delete_count += 1

    print('Deleted {} image(s) and {} xml file(s)'.format(img_delete_count, xml_delete_count))


def getFiles(directory='', fileType='.jpg', startWithName=''):
    res = []
    directory = os.getcwd() + '/' + directory
    for currentfile in os.listdir(directory):
        if currentfile.lower().startswith(startWithName.lower()) and currentfile.endswith(fileType):
            res.append(os.path.join(directory, currentfile))

    return sorted(res)

def findFile(name, directory=''):
    directory = os.getcwd() + '/' + directory
    for currentfile in os.listdir(directory):
        if currentfile.lower() == name.lower():
            return os.path.join(directory, currentfile)

    return None

if __name__ == "__main__":
    main()
