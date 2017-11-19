import tensorflow as tf
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 20

n_classes = 32

x = tf.placeholder(tf.float32, [None, 128, 128, 3])
y = tf.placeholder(tf.float32, [None, n_classes])

keep_prob = tf.placeholder(tf.float32)

def read_wrl(name_file, verbosity=0):
    holder = []
    with open(name_file, 'r') as vrml:
        k = 0
        for lines in vrml:
            if 'coordIndex' in lines:
                print 'BREAK'
                break
            vert = re.findall("[-0-9]{1,3}.[0-9]{3}", lines)
            k += 1
            if len(vert) == 3 and k <7900:
                print 'BREAK2'
                if all('.' in vert[i] for i in range(0,3)):
                    holder.append(map(float, vert))
    holder_array = np.array(holder)

    if verbosity == 1:
        x, y, z = zip(*holder)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot(x, y, z)
        plt.show()
        import ipdb;ipdb.set_trace()
    return holder_array
class read_Images(object):

    def __init__(self, path, data_type):
        self.data_path = path
        self.type = data_type

    def read_Images(self):
        if self.type == '3d':
            wrl_per_folder = glob.glob(self.data_path+'/'+'*.wrl')
            for i_wrl in wrl_per_folder:
                d_face = read_wrl(name_file=i_wrl, verbosity=1)
                import ipdb;ipdb.set_trace()



IMAGE = read_Images(path='/media/iman/Elements/Database/BU_3DFE/F0001', data_type='3d')
all = IMAGE.read_Images()
import ipdb;ipdb.set_trace()
class TensorFunctions(object):

    def conv2d(self, name, l_input, n_filter, stride_size, kernel):
        return tf.nn.relu(tf.layers.conv2d(l_input, filters=n_filter,kernel_size=[1, kernel, kernel, 1],
                                           strides=[1, stride_size, stride_size, 1], padding='SAME', name=name))
    def max_pool(self, name, l_input, k):
        return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

    def norm(self, name, l_input, lsize=4):
        return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def model(_X, n_input):
    _X = tf.reshape(_X, shape=[-1, n_input])
    conv1 = TensorFunctions.conv2d(l_input=_X, n_filter=128, stride_size=3, kernel=3, name='Convolution_1')
    return conv1



