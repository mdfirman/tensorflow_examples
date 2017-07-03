###########################################
# Simple tensorflow 3-class classification

# Imports
import tensorflow as tf
layers = tf.contrib.slim.layers
import tensorflow.contrib.slim as slim
import numpy as np

###########################################
# ## Setting variables and parameters

num_classes = 3  # three-class classification
learning_rate = 0.001
num_epochs = 20
batch_size = 64

###########################################
# ## Loading data
#
# Here, we're using mnist as example data. We filter the data here to just select three classes of images.

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def load_mnist(data_split):
    """This function extract mnist images and labels, but only keeps thefirst num_classes classes"""
    y = np.argmax(data_split.labels, axis=1)
    to_keep = y < num_classes
    return data_split.images[to_keep], data_split.labels[to_keep, :num_classes]


def load_data():
    """Load mnist train, validation and test data"""
    train_x, train_y = load_mnist(mnist.train)
    val_x, val_y = load_mnist(mnist.validation)
    test_x, test_y = load_mnist(mnist.test)

    return train_x, train_y, val_x, val_y, test_x, test_y


train_x, train_y, val_x, val_y, test_x, test_y = load_data()


###########################################
# ## Defining tf variables and creating network
#
# This is where the network architecture is defined.


# Network inputs. Assuming input images are 28x28x1
x_in = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
y_in = tf.placeholder(tf.float32, shape=(None, num_classes))
phase = tf.placeholder(tf.bool)  # for batch norm - see http://ruishu.io/2016/12/27/batchnorm/

# We are using scoping to make the code cleaner - e.g. each activation layer
# has a relu, so let's only say that once
activation_layers = [layers.conv2d, layers.fully_connected]

with tf.contrib.slim.arg_scope(activation_layers, activation_fn=tf.nn.relu):
    with tf.contrib.slim.arg_scope([layers.batch_norm], center=True, scale=True, is_training=phase):

        enc = layers.conv2d(x_in, 16, 3)
        enc = layers.max_pool2d(enc, 2, 2)
        enc = layers.batch_norm(enc)

        enc = layers.conv2d(enc, 32, 3)
        enc = layers.max_pool2d(enc, 2, 2)
        enc = layers.batch_norm(enc)

        enc = layers.conv2d(enc, 64, 3)
        enc = layers.max_pool2d(enc, 2, 2)
        enc = layers.batch_norm(enc)

        bsize, h, w, c = enc.shape.as_list()

        # dense layers
        dense = tf.reshape(enc, [-1, h*w*c])
        dense = layers.fully_connected(dense, 128)
        dense = layers.batch_norm(dense)

        dense = layers.fully_connected(dense, 128)
        output = layers.fully_connected(dense, num_classes, activation_fn=None)

###########################################
# ## Defining losses and updates

# We let tensorflow do the sigmoid nonlinearity for us
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_in, logits=output))

# This defines the function which will update the network weights
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = slim.learning.create_train_op(loss, optimizer)

# Some code to give us the accuracy
labels = tf.argmax(y_in, axis=1)
predictions = tf.argmax(output, axis=1)
accuracy = tf.contrib.metrics.accuracy(labels=labels, predictions=predictions)


# ## Training model

def get_feed_dict(x_data, y_data, batch_size):
    """A helper function to generate minibatches from training
    OR validation data"""

    for start_idx in range(0, len(x_data), batch_size):
        end_idx = min(len(x_data), start_idx + batch_size)

        x = x_data[start_idx:end_idx].reshape(-1, 28, 28, 1)
        y = y_data[start_idx:end_idx]

        yield {y_in: y, x_in: x}


# Do training here... make sure to pass phase=True in the feed_dict for training, phase=False for testing
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for epoch in range(num_epochs):

    print "Epoch: ", epoch

    #####################
    # TRAINING

    all_train_losses = []
    all_train_accuracies = []

    for feed_dictionary in get_feed_dict(train_x, train_y, batch_size):

        feed_dictionary[phase] = True
        _, trn_loss, trn_acc = sess.run([train_step, loss, accuracy], feed_dictionary)

        all_train_losses.append(trn_loss)
        all_train_accuracies.append(trn_acc)

    print "-- Training loss: %0.5f" % np.mean(all_train_losses)
    print "-- Training accuracy: %0.5f" % np.mean(all_train_accuracies)

    #####################
    # VALIDATION

    all_validation_losses = []
    all_validation_accuracies = []

    for feed_dictionary in get_feed_dict(val_x, val_y, batch_size):

        feed_dictionary[phase] = False
        val_loss, val_acc = sess.run([loss, accuracy], feed_dictionary)

        all_validation_losses.append(val_loss)
        all_validation_accuracies.append(val_acc)

    print "-- Validation loss: %0.5f" % np.mean(all_validation_losses)
    print "-- Validation accuracy: %0.5f" % np.mean(all_validation_accuracies)
