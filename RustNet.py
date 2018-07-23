
# coding: utf-8

# In[12]:


import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import transform
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


dir_path = os.getcwd() + '\images'
print(dir_path)


# In[14]:


def transformData(path):
    data = []
    for filename in os.listdir(path):
        if filename.endswith('.PNG'):
            image = plt.imread(path + '\\' + filename, 0)
            x = transform.resize(image, (64,64))
            data.append(x)
    return np.array(data)


# In[15]:


def getLabels(filename):
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            l = line.split(',')
            l = l[-1].strip()
            labels.append(l)
    return np.array(labels)


# In[16]:


fig = plt.figure(figsize = (10, 10))
rows = 5
cols = 5
counter = 1
for row in range(rows):
    for col in range(cols):
        plt.subplot(rows, cols, counter)
        plt.axis('off')
        plt.imshow(data[counter])
        counter += 1
plt.show()


# In[98]:


def model(batch_image, phase):

    assert batch_image.get_shape().as_list() == [None, 64, 64,3]
    
    with tf.variable_scope('Convlayer1'):
        out1 = tf.layers.conv2d(batch_image, filters = 16,kernel_size = 3, padding ='same', name = 'Conv1')
        #outbn1 = tf.layers.batch_normalization(out1, training = phase, name = 'BatchNorm1')
        out1 = tf.nn.relu(out1, name = 'Relu1')
        out1 = tf.layers.max_pooling2d(out1,pool_size = 2 , strides = 2, name = 'Maxpool1')
    with tf.variable_scope('Convlayer2'):
        out2 = tf.layers.conv2d(out1, filters = 32,kernel_size = 3, padding ='same', name = 'Conv2' )
        #outbn2 = tf.layers.batch_normalization(out2, training = phase, name = 'BatchNorm2')
        out2 = tf.nn.relu(out2, name = 'Relu2')
        out2 = tf.layers.max_pooling2d(out2,pool_size = 2 , strides = 2, name = 'Maxpool2')
    with tf.variable_scope('Convlayer3'):
        out3 = tf.layers.conv2d(out2, filters = 64,kernel_size = 3, padding ='same', name = 'Conv3' )
        #outbn3 = tf.layers.batch_normalization(out3, training = phase, name = 'BatchNorm3')
        out3 = tf.nn.relu(out3, name = 'Relu3')
        out3 = tf.layers.max_pooling2d(out3,pool_size = 2 , strides = 2, name = 'Maxpool3')
    with tf.variable_scope('Convlayer4'):
        out4 = tf.layers.conv2d(out3, filters = 128,kernel_size = 3, padding ='same', name = 'Conv4' )
        #outbn4 = tf.layers.batch_normalization(out4, training = phase, name = 'BatchNorm4')
        out4 = tf.nn.relu(out4, name = 'Relu4')
        out4 = tf.layers.max_pooling2d(out4,pool_size = 2 , strides = 2, name = 'Maxpool4')
    
    assert out4.get_shape().as_list() == [None, 4, 4, 128]
    
    out4 = tf.reshape(out4, shape =[-1, 4*4*128 ])
    
    with tf.variable_scope('FClayerl'):
        out5 = tf.layers.dense(out4, units = 128, name = 'fc1')
        #outbn5 = tf.layers.batch_normalization(out5, training = phase, name = 'BatchNorm4')
        out5 = tf.nn.relu(out5)
    with tf.variable_scope('FClayer2'):
        out6 = tf.layers.dense(out5, units = 3, name = 'fc2')

    return out6


# In[99]:


def train(batch_image, labels, phase):

    logit= model(batch_image, phase)
    true_labels = labels
    true_labels= tf.cast(true_labels, dtype = tf.int64)
    with tf.variable_scope('Loss'):
        loss = tf.losses.sparse_softmax_cross_entropy(labels = true_labels, logits = logit)
    optimizer = tf.train.AdamOptimizer()
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss)
    predictions = tf.argmax(logit, 1)
    with tf.variable_scope('Accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(true_labels, predictions), tf.float32))



    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('train_image', batch_image)
    model_spec = {}
    model_spec['loss']  = loss
    model_spec['train_op'] = train_op
    model_spec['accuracy'] = accuracy
    model_spec['summary_op'] = tf.summary.merge_all()

    return model_spec


# In[106]:


if __name__ == '__main__':
    tf.reset_default_graph()
    data = transformData(dir_path)
    labels = getLabels('Labels.txt')
    np.random.seed(seed = 2018)
    index = np.random.permutation(len(data))
    data = data[index]
    labels = labels[index]
    with tf.variable_scope('Placeholders'):
        X = tf.placeholder('float',shape=[None,64,64,3],name='image')
        Y = tf.placeholder('int32', shape=[None], name='labels')
        batch_size = tf.placeholder(dtype = tf.int64)
        phase = tf.placeholder(dtype = tf.bool, name='is_training') 



    model_spec = train(X , Y, phase)

    loss = model_spec['loss']
    train_op = model_spec['train_op']
    accuracy = model_spec['accuracy']
    summary = model_spec['summary_op']
    BATCH_SIZE = 32
    EPOCHS = 30
    
    n_batches = (len(data) + BATCH_SIZE - 1)// BATCH_SIZE
    global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
    increment_global_step_op = tf.assign(global_step, global_step+1)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #summary_writer = tf.summary.FileWriter('sign_classifier/tensorboardlogs/run10_placeholder/', sess.graph)
        #saver = tf.train.Saver()
        print('Training...')
        for i in range(EPOCHS):
            # initialise iterator with train data
            tot_loss = 0
            end = 0    
            for n in range(n_batches):
                start = end
                if (end+ BATCH_SIZE < len(data)):    
                    end = end + BATCH_SIZE
                else:
                    end = len(data)
                image_batch = data[start:end ]
                label_batch = labels[start:end]
                
                _, loss_value ,step= sess.run([train_op, loss, increment_global_step_op], feed_dict = {X: image_batch, Y: label_batch, batch_size: BATCH_SIZE, phase: True})
                tot_loss += loss_value
                print('Epoch: {}, Step: {} completed'.format(i, step))
                
                if step % 100 == 0:
                    saver.save(sess, 'sign_classifier/checkpoints_ph/sign-classifier', global_step = step)

            print("Iter: {}, Loss: {:.4f}".format(i, tot_loss))

