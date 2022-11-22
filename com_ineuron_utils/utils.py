import base64
import os

def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    os.makedirs(os.path.join('test'),exist_ok=True)
    with open("test/"+fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())