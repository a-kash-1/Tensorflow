
# coding: utf-8

# In[2]:
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import transform


# In[3]:
tf.reset_default_graph()
saver = tf.train.import_meta_graph('RustNet\\checkpoints-500.meta')
g = tf.get_default_graph()
X = g.get_operation_by_name('Placeholders/image').outputs[0]
Y = g.get_operation_by_name('Placeholders/labels').outputs[0]
batch_size = g.get_operation_by_name('Placeholders/Placeholder').outputs[0]
phase = g.get_operation_by_name('Placeholders/is_training').outputs[0]
output = g.get_operation_by_name('FClayer2/fc2/BiasAdd').outputs[0]
inputs = g.get_operation_by_name('Convlayer1/Conv1/Conv2D').inputs[0]


# In[35]:
dir_path = os.getcwd() + '\\test'
print(dir_path)


# In[40]:
def transformData(path):
    data = []
    for filename in os.listdir(path):
        if filename.endswith('.PNG'):
            image = plt.imread(path + '\\' + filename, 0)
            x = transform.resize(image, (64,64))
            data.append(x)          
    return np.array(data)


# In[41]:
data = transformData(dir_path)


# In[39]:
#prediction on nine random images from datset
_data = data[:9]
#_labels = labels[:9]
with tf.Session() as sess:
    
    saver.restore(sess, tf.train.latest_checkpoint('RustNet\\'))
    logit, _image = sess.run([output, inputs], feed_dict = {X: _data, batch_size: len(_data), phase: False})
    print('Predicted Label: {}'.format(np.argmax(logit, axis = 1)))

