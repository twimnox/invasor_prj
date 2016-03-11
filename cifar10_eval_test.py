"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""


#did i just change this ?!?!?!
from datetime import datetime
import math
import time
from sklearn.feature_extraction.tests import test_image

import cv2
import os

import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf

#from tensorflow.models.image.cifar10 import cifar10
import cifar10
import cifar10_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/prtricardo/tensorflow_tmp/acacia10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/prtricardo/tensorflow_tmp/acacia10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 1, #era 10000
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('test_dir', '/home/prtricardo/tensorflow_tmp/acacia10_test',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('test_file', 'mstar_test_batch.bin',
                           """Name of test file.""")
tf.app.flags.DEFINE_integer('image_size', 24,
                            """Size of image.""")
tf.app.flags.DEFINE_integer('num_classes', 3,
                            """Number of classes.""")
tf.app.flags.DEFINE_integer('batch_size_here', 1,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_channels', 3,
                            """Number of channels""")


def readd_cifar10(filename_queue):
  """Reads and parses examples from CIFAR10 data files.
  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.
  Args:
    filename_queue: A queue of strings with the filenames to read from.
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 1  # 2 for CIFAR-100
  result.height = FLAGS.image_size
  result.width = FLAGS.image_size
  result.depth = FLAGS.num_channels
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                           [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result


def inputs():
  filenames = [os.path.join(FLAGS.test_dir, FLAGS.test_file)]
  for f in filenames:
    if not gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)
  filename_queue = tf.train.string_input_producer(filenames,shuffle=False)
  read_input = readd_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  height = FLAGS.image_size
  width = FLAGS.image_size
  float_image = tf.image.per_image_whitening(reshaped_image)
  num_preprocess_threads = 1
  images, label_batch = tf.train.batch(
      [float_image, read_input.label],
      batch_size=FLAGS.batch_size_here,
      num_threads=num_preprocess_threads,
      capacity=FLAGS.batch_size_here)
  tf.image_summary('images', images, max_images = 29)
  print "label batch @inputs()", read_input.label
  return images, tf.reshape(label_batch, [FLAGS.batch_size_here])


def eval_once(saver, summary_writer, top_k_op, top_k_predict_op, summary_op, images):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print 'No checkpoint file found'
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size_here))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size_here
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        image, test_labels = sess.run([images, top_k_predict_op])
        classification = sess.run(top_k_predict_op)
        print (step, int(test_labels[0]))
        print "network predicted:", classification[0], "for real label:", test_labels
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = float(true_count) / float(total_sample_count)
      print '%s: precision @ 1 = %.3f' % (datetime.now(), precision)



      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception, e:  # pylint: disable=broad-except
      coord.request_stop(e)
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def test_image(saver, logits):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """


  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print 'No checkpoint file found'
      return

    width = 32
    height = 32
    images = np.zeros((1, 3072))
    path = "/home/prtricardo/tese_ws/open_cv/acacia_model/acacia_.jpg"
    img = cv2.imread(path)
    resized_image = cv2.resize(img, (width, height))
    arr = np.uint8([0 for x in range(width*height*3)])


    arr_cnt = 0
    for y in range(0, width):
      for x in range(0, height):
          arr[arr_cnt] = np.uint8(resized_image[x, y, 2])  # R
          arr[arr_cnt + 1024] = np.uint8(resized_image[x, y, 1])  # G
          arr[arr_cnt + 2048] = np.uint8(resized_image[x, y, 0])  # B

          arr_cnt += 1



    image = arr




    #images[0] = arr

    # filename_queue = tf.train.string_input_producer("/home/prtricardo/tese_ws/open_cv/acacia_model/acacia.bin")
    #
    # # Read examples from files in the filename queue.
    # read_input = cifar10_input.read_cifar10(filename_queue)
    # reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    # ima = tf.cast(images[0], np.int32)

    # classification = sess.run(tf.argmax(logits, 1), feed_dict={reshaped_image})
    # classification = sess.run(tf.argmax(logits, 1), feed_dict={x: [images[0]]})
    classification = sess.run(tf.argmax(logits, 1))

    #classification = sess.run(logits)
    print classification


    print 'Neural Network predicted', classification[0]



def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    images, labels = inputs() # cifar10.inputs(eval_data=eval_data)
    print "images dimension @ evaluation:", images
    print "image label @ evaluation:", labels



    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)



    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    top_k_predict_op = tf.argmax(logits, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = {}
    for v in tf.all_variables():
      if v in tf.trainable_variables():
        restore_name = variable_averages.average_name(v)
      else:
        restore_name = v.op.name
      variables_to_restore[restore_name] = v
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir,
                                            graph_def=graph_def)

    # test_image(saver, logits)

    ###Test one image
    print "#################"
    print "##gonna test this:"
    print "#################"




    ###

    while True:
      eval_once(saver, summary_writer, top_k_op, top_k_predict_op, summary_op, images)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  # cifar10.maybe_download_and_extract()
  # if gfile.Exists(FLAGS.eval_dir):
  #   gfile.DeleteRecursively(FLAGS.eval_dir)
  # gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
