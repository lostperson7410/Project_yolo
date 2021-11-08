from PIL import Image
import imagehash
import hashlib

from numpy.lib.function_base import average
image_file = open('./data/check140.jpg', 'rb').read()
md5hash = hashlib.md5(image_file).hexdigest()
print(md5hash)

image_file = Image.open('./data/check140.jpg')
phashONE = imagehash.phash(image_file)
print(phashONE)
a = str(phashONE)

image_file = Image.open('./data/check127.jpg')
phashTWO = imagehash.phash(image_file)
print(phashTWO)
b = str(phashTWO)

gs_hash = imagehash.hex_to_hash(a)
ori_hash = imagehash.hex_to_hash(b)
avg_hash = gs_hash - ori_hash
print('Hamming distance:', gs_hash - ori_hash)

if avg_hash <= 22 :
    print('image is similar')
else : print('image is identical')