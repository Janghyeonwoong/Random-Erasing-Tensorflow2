import tensorflow as tf
import os
import erasing
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 5)

img_dir    = './data'
result_dir = './result'

img_width    = 300
img_height   = 200
img_channels = 3

def dataset_map_function(path):
    path  = tf.io.read_file(path)
    img = tf.io.decode_jpeg(path, channels=img_channels)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    # image 3D Tensor (None, None, 3)
    img = tf.image.resize(img, [img_height, img_width])
    # image 3D Tesnor (img_height, img_width, img_channels)
    
    img = erasing.random_erasing(img)
    return img

def visualize(img1, img2, num):
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.title('image 1')
    plt.imshow(img1)

    plt.subplot(1,2,2)
    plt.title('image 2')
    plt.imshow(img2)
    plt.savefig(os.path.join(result_dir, str(num)))


if __name__ == '__main__':
    img_list = os.listdir(img_dir)
    joined_img_list = [os.path.join(img_dir, path) for path in img_list]

    dataset = tf.data.Dataset.from_tensor_slices(joined_img_list)
    dataset = dataset.map(dataset_map_function)
    dataset = dataset.batch(batch_size=2)
    result  = list(dataset.as_numpy_iterator())
    for index in range(len(result)):
        visualize(result[index][0], result[index][1], index)

    print('Complete')
