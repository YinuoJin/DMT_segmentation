import sys
from matplotlib import image as mpimg
import numpy as np
import os

DIPHA_CONST = 8067171840
DIPHA_IMAGE_TYPE_CONST = 1
DIM = 3

# Modify input as single-image filename
"""
input_dir = os.path.join(os.getcwd(), sys.argv[1])
dipha_output_filename = sys.argv[2]
vert_filename = sys.argv[3]

input_filenames = [name
                   for name in os.listdir(input_dir)
                   if (os.path.isfile(input_dir + '/' + name)) and (name != ".DS_Store")]
input_filenames.sort()

iamge = mpimg.imread(os.path.join(input_dir, input_filenames[0]))
"""
input_filename = os.path.join(os.getcwd(), sys.argv[1])
dipha_output_filename = sys.argv[2]
vert_filename = sys.argv[3]

assert os.path.exists(input_filename) and os.path.isfile(input_filename), "Input image file {} doesn't exist".format(input_filename)

image = mpimg.imread(input_filename)
nx, ny = image.shape
del image


# nz = len(input_filenames)
nz = 1

print(nx, ny, nz)
#sys.exit()
im_cube = np.zeros([nx, ny, nz])

i = 0
"""
for name in input_filename:
    sys.stdout.flush()
    print(i, name)
    fileName = input_dir + "/" + name
    im_cube[:, :, i] = mpimg.imread(fileName)
    i = i + 1
"""
sys.stdout.flush()
print(i, input_filename)
im_cube[:, :, 0] = mpimg.imread(input_filename)

print('writing dipha output...')
with open(dipha_output_filename, 'wb') as output_file:
    # this is needed to verify you are giving dipha a dipha file
    np.int64(DIPHA_CONST).tofile(output_file)
    # this tells dipha that we are giving an image as input
    np.int64(DIPHA_IMAGE_TYPE_CONST).tofile(output_file)
    # number of points
    np.int64(nx * ny * nz).tofile(output_file)
    # dimension
    np.int64(DIM).tofile(output_file)
    # pixels in each dimension
    np.int64(nx).tofile(output_file)
    np.int64(ny).tofile(output_file)
    np.int64(nz).tofile(output_file)
    # pixel values
    for k in range(nz):
        sys.stdout.flush()
        print('dipha - working on image', k)
        for j in range(ny):
            for i in range(nx):
                val = int(-im_cube[i, j, k]*255)
                '''
                if val != 0 and val != -1:
                    print('val check:', val)
                '''
                np.float64(val).tofile(output_file)
    output_file.close()

print('writing vert file')
with open(vert_filename, 'w') as vert_file:
    for k in range(nz):
        sys.stdout.flush()
        print('verts - working on image', k)
        for j in range(ny):
            for i in range(nx):
                vert_file.write(str(i) + ' ' + str(j) + ' ' + str(k) + ' ' + str(int(-im_cube[i, j, k] * 255)) + '\n')
    vert_file.close()

print(nx, ny, nz)
