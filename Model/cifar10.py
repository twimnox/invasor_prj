"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use input() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
#this is not a very stable version yet
# pylint: disable=missing-docstring
import gzip
import os
import re
import sys
import tarfile
import urllib

import tensorflow.python.platform
import tensorflow as tf

#from tensorflow.models.image.cifar10 import cifar10_input
import cifar10_input

from tensorflow.python.platform import gfile

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 1, #era 128 @TODO tem de ser sempre 1
                            """Number of images to process in a batch.""")
# tf.app.flags.DEFINE_string('data_dir', '/home/prtricardo/tensorflow_tmp/200x200_models/acacia10_data',
#                            """Path to the CIFAR-10 data directory.""")
# tf.app.flags.DEFINE_integer('dataset_window_size', 200,
#                             """"Window size of the original images""")
# tf.app.flags.DEFINE_integer('noise_crop_size', 180,
#                             """"noise crop size of the original images for distortion toleration""")


# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
# IMAGE_SIZE = FLAGS.noise_crop_size

# Global constants describing the CIFAR-10 data set.
# NUM_CLASSES = 4 #era 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2400 #era 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 668 #era 10000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # era 0.9999The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 100   #era 350   # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.02  # era 0.1 Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.02 #era 0.1       # Initial learning rate.
MIN_FRACTION_OF_EXAMPLES_QUEUE = 0.4

# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def put_kernels_on_grid (kernel, (grid_Y, grid_X), pad=1):
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
    '''
    # pad X and Y
    x1 = tf.pad(kernel, tf.constant( [[pad,0],[pad,0],[0,0],[0,0]] ))

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + pad
    X = kernel.get_shape()[1] + pad

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, 3]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, 3]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 1]
    x_min = tf.reduce_min(x7)
    x_max = tf.reduce_max(x7)
    x8 = (x7 - x_min) / (x_max - x_min)

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x8, dtype=tf.uint8)



def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name) # era [0-9]
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def _generate_image_and_label_batch(image, label, min_queue_examples):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [IMAGE_SIZE, IMAGE_SIZE, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'FLAGS.batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  images, label_batch = tf.train.shuffle_batch(
      [image, label],
      batch_size=FLAGS.batch_size,
      num_threads=num_preprocess_threads,
      capacity=min_queue_examples + 3 * FLAGS.batch_size,
      min_after_dequeue=min_queue_examples)

  # Display the training images in the visualizer.
  tf.image_summary('images', images)

  return images, tf.reshape(label_batch, [FLAGS.batch_size])


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Raises:
    ValueError: if no data_dir

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = [os.path.join(FLAGS.data_dir, 'acacia-10-batches-bin',
                            'data_batch_%d.bin' % i)
               for i in xrange(0, 4)]
  for f in filenames:
    if not gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = cifar10_input.read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly crop a [height, width] section of the image.
  distorted_image = tf.image.random_crop(reshaped_image, [height, width])

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Because these operations are not commutative, consider randomizing
  # randomize the order their operation.
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=20) #era 63
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.7, upper=1.1) #era 0.2 e 1.8

  # distorted_image = reshaped_image

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(distorted_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = MIN_FRACTION_OF_EXAMPLES_QUEUE # 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with LALALALALu %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  print "I have read", read_input.label, "for this batch"
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples)


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Raises:
    ValueError: if no data_dir

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')

  if not eval_data:
    filenames = [os.path.join(FLAGS.data_dir, 'acacia-10-batches-bin',
                              'data_batch_%d.bin' % i)
                 for i in xrange(0, 3)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(FLAGS.data_dir, 'acacia-10-batches-bin',
                              'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = cifar10_input.read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         width, height)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(resized_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples)


def inference(images, NUM_CLASSES):

  print "inference n_classes: ", NUM_CLASSES
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[7, 7, 3, 64], #era 5 5 3 64
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

    #visualize 3 random filters from conv layer 1
        # with tf.variable_scope('visualization'):
        #     # scale weights to [0 255] and convert to uint8 (maybe change scaling?)
        #     x_min = tf.reduce_min(kernel)
        #     x_max = tf.reduce_max(kernel)
        #     kernel_0_to_1 = (kernel - x_min) / (x_max - x_min)
        #     kernel_0_to_255_uint8 = tf.cast(kernel_0_to_1, dtype=tf.float32)#tf.image.convert_image_dtype(kernel_0_to_1, dtype=tf.uint8)
        #
        #     # to tf.image_summary format [batch_size, height, width, channels]
        #     kernel_transposed = tf.transpose (kernel_0_to_255_uint8, [3, 0, 1, 2])
        #
        #     # this will display random 3 filters from the 64 in conv1
        #     tf.image_summary('conv1/filters', kernel_transposed, max_images=3)


    # # Visualize conv1 features
    # with tf.variable_scope('conv1') as scope_conv:
    #     #tf.get_variable_scope().reuse_variables()
    #     scope_conv.reuse_variables()
    #     weights = tf.get_variable('weights')
    #     grid_x = grid_y = 8   # to get a square grid for 64 conv1 features
    #     grid = put_kernels_on_grid (weights, (grid_y, grid_x))
    #     tf.image_summary('conv1/features', grid, max_images=1)



  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 5, 5, 1], strides=[1, 4, 4, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64], #era 5 5 64 64
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 10, 10, 1],
                         strides=[1, 4, 4, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    dim = 1
    for d in pool2.get_shape()[1:].as_list():
      dim *= d
    reshape = tf.reshape(pool2, [FLAGS.batch_size, dim])

    weights = _variable_with_weight_decay('weights', shape=[dim, 384], #era 384
                                          stddev=0.04, wd=0.004) #wd era 0.04
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1)) #era 384
    local3 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192], #era 384, 192
                                          stddev=0.04, wd=0.004) #wd era 0.04
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1)) #era 192
    local4 = tf.nn.relu_layer(local3, weights, biases, name=scope.name)
    _activation_summary(local4)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES], #era 192
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.nn.xw_plus_b(local4, weights, biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Reshape the labels into a dense Tensor of
  # shape [batch_size, NUM_CLASSES].
  sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
  indices = tf.reshape(tf.range(0, FLAGS.batch_size, 1), [FLAGS.batch_size, 1])
  concated = tf.concat(1, [indices, sparse_labels])
  dense_labels = tf.sparse_to_dense(concated,
                                    [FLAGS.batch_size, NUM_CLASSES],
                                    1.0, 0.0)

  # Calculate the average cross entropy loss across the batch.
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      logits, dense_labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')





def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summmary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op



def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """






  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')


  return train_op


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
    print
    statinfo = os.stat(filepath)
    print 'Succesfully downloaded', filename, statinfo.st_size, 'bytes.'
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
