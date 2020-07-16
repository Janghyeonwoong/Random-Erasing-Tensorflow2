import tensorflow as tf
import os
import erasing
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse

mpl.rcParams['figure.figsize'] = (12, 5)

img_dir    = './data'
result_dir = './result'

img_width    = 300
img_height   = 200
img_channels = 3

parser = argparse.ArgumentParser(description='Random Erasing Tensorflow2')
parser.add_argument('-m', '--method', default='random', type=str, help='choose method (white, black, random)')
parser.add_argument('-c', '--channels', default=3, type=int, help='Image\'s channel')
parser.add_argument('-W', '--Width', default=300, type=int, help='Image\'s width')
parser.add_argument('-H', '--Height', default=200, type=int, help='Image\'s height')
parser.add_argument('-i','--img_dir', default='./data', type=str, help='Image directory')
parser.add_argument('-r','--result_dir', default='./result', type=str, help='Save result directory')

args = parser.parse_args()

def dataset_map_function(path):
    path  = tf.io.read_file(path)
    img = tf.io.decode_jpeg(path, channels=img_channels)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    # image 3D Tensor (None, None, 3)
    img = tf.image.resize(img, [args.Height, args.Width])
    # image 3D Tesnor (img_height, img_width, img_channels)
    
    img = erasing.random_erasing(img, method=args.method)
    return img
    
def visualize(img1, img2, num):
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.title('image 1')
    plt.imshow(img1)

    plt.subplot(1,2,2)
    plt.title('image 2')
    plt.imshow(img2)
    plt.savefig(os.path.join(args.result_dir, str(num)))


if __name__ == '__main__':
    
    img_list = os.listdir(args.img_dir)
    joined_img_list = [os.path.join(args.img_dir, path) for path in img_list]

    dataset = tf.data.Dataset.from_tensor_slices(joined_img_list)
    dataset = dataset.map(dataset_map_function)
    dataset = dataset.batch(batch_size=2)
    result  = list(dataset.as_numpy_iterator())
    for index in range(len(result)):
        visualize(result[index][0], result[index][1], index)

    print('Complete')