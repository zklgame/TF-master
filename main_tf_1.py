import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from config import Config
import argparse
import time
from tqdm import tqdm
tf.disable_eager_execution()


config = Config(output_dir='outputs/v1', epochs=10, batch_size=32, log_every_step=100, v1=True)


def define_cnn(x, n_classes, reuse, is_training):
    with tf.variable_scope('cnn', reuse=reuse):
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(conv1, 63, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        shape = (-1, conv2.shape[1] * conv2.shape[2] * conv2.shape[3])
        fc1 = tf.reshape(conv2, shape)
        fc1 = tf.layers.dense(fc1, 1024)
        fc1 = tf.layers.dropout(fc1, rate=0.5, training=is_training)

        out = tf.layers.dense(fc1, n_classes)

    return out


def load_data():
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

    # Scale input in [-1, 1] range
    train_x = train_x / 255. * 2 - 1
    test_x = test_x / 255. * 2 - 1

    # Add the last 1 dimension, so to have images 28x28x1
    train_x = np.expand_dims(train_x, -1)
    test_x = np.expand_dims(test_x, -1)

    return train_x, train_y, test_x, test_y


def main(is_training=False):
    batch_size = config.batch_size

    # 1. load data
    if is_training:
        train_x, train_y, val_x, val_y = load_data()
    else:
        _, _, test_x, test_y = load_data()

    # 2. define model
    input = tf.placeholder(tf.float32, (None, 28, 28, 1))
    labels = tf.placeholder(tf.int64, (None, ))
    logits = define_cnn(input, n_classes=10, reuse=False, is_training=True)

    # 3. define loss_op, global_step_op, train_op, predictions_op, accuracy_op, val_accuracy_op, savers
    loss_op = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    global_step_op = tf.train.get_or_create_global_step()
    train_op = tf.train.AdamOptimizer().minimize(loss_op, global_step_op)

    predictions_op = tf.argmax(logits, axis=1)
    correct_predictions_op = tf.equal(predictions_op, labels)
    accuracy_op = tf.reduce_sum(
        tf.cast(correct_predictions_op, tf.float32), name='accuracy'
    )

    val_accuracy_op = tf.Variable(0.0, name='val_accuracy', dtype=tf.float32)
    train_saver = tf.train.Saver(max_to_keep=3)
    val_saver = tf.train.Saver(max_to_keep=3)

    # 4. define summary writer
    train_summary_writer = tf.summary.FileWriter(config.log_dir_train, tf.get_default_graph())
    validation_summary_writer = tf.summary.FileWriter(config.log_dir_dev, tf.get_default_graph())
    accuracy_summary_op = tf.summary.scalar('accuracy', accuracy_op)
    loss_summary_op = tf.summary.scalar('loss', loss_op)

    # 5. start session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        if is_training:
            latest_ckpt = tf.train.latest_checkpoint(config.ckpt_dir_train)
            if latest_ckpt:
                train_saver.restore(sess, latest_ckpt)
                print(f"Restored from {latest_ckpt}")
            else:
                sess.run(tf.global_variables_initializer())
                print("Initializing from scratch.")
        else:
            latest_ckpt = tf.train.latest_checkpoint(config.ckpt_dir_dev)
            if latest_ckpt:
                val_saver.restore(sess, latest_ckpt)
                print(f"Restored from {latest_ckpt}")
            else:
                sess.run(tf.global_variables_initializer())
                print("Initializing from scratch.")

        print(f'@zkl start from step {sess.run(global_step_op)} with val_accuracy {sess.run(val_accuracy_op)}')

        if not is_training:
            loss = 0.0
            accuracy = 0.0
            for t in tqdm(range(int((test_x.shape[0] - 1) / batch_size) + 1)):
                start_from = t * batch_size
                to = (t + 1) * batch_size
                loss_val, accuracy_val = sess.run([loss_op, accuracy_op], feed_dict={
                    input: test_x[start_from: to],
                    labels: test_y[start_from: to]
                })
                loss += loss_val
                accuracy += accuracy_val

            loss /= test_x.shape[0]
            accuracy /= test_x.shape[0]
            print(f"Test accuracy: {accuracy}, loss: {loss}")
            return

        # 6. real training
        nr_batches_train = int(train_x.shape[0] / batch_size)
        print(f"Number of batches per epoch: {nr_batches_train}")
        for epoch in range(config.epochs):
            global_step = sess.run(global_step_op)
            np.random.seed(global_step)
            np.random.shuffle(train_x)
            np.random.seed(global_step)
            np.random.shuffle(train_y)

            mean_loss = 0.0
            mean_accuracy = 0.0

            start_time = time.time()
            for t in range(nr_batches_train):
                start_from = t * batch_size
                to = (t + 1) * batch_size

                loss_value, accuracy_value, _ = sess.run([loss_op, accuracy_op, train_op], feed_dict={
                    input: train_x[start_from: to],
                    labels: train_y[start_from: to]
                })

                mean_loss += loss_value
                mean_accuracy += accuracy_value

                if t and t % config.log_every_step == 0:
                    mean_loss /= config.log_every_step * config.batch_size
                    mean_accuracy /= config.log_every_step * config.batch_size

                    global_step, loss_summary, accuracy_summary = sess.run([global_step_op, loss_summary_op, accuracy_summary_op], feed_dict={
                        input: train_x[start_from: to],
                        labels: train_y[start_from: to]
                    })

                    print(f"{global_step}: {mean_loss} - accuracy: {mean_accuracy} - time: {time.time() - start_time}")
                    save_path = train_saver.save(sess, config.ckpt_train, global_step=global_step)
                    print(f"Checkpoint saved: {save_path}")

                    train_summary_writer.add_summary(loss_summary, global_step)
                    train_summary_writer.add_summary(accuracy_summary, global_step)

                    mean_loss = 0.0
                    mean_accuracy = 0.0

                    print('')
                    start_time = time.time()

            print(f"Epoch {epoch} terminated")

            # Measuring accuracy on the whole validation set at the end of the epoch
            loss = 0.0
            accuracy = 0.0
            for t in tqdm(range(int((val_x.shape[0] - 1) / batch_size) + 1)):
                start_from = t * batch_size
                to = (t + 1) * batch_size
                loss_val, accuracy_val = sess.run([loss_op, accuracy_op], feed_dict={
                    input: val_x[start_from: to],
                    labels: val_y[start_from: to]
                })
                loss += loss_val
                accuracy += accuracy_val

            loss /= val_x.shape[0]
            accuracy /= val_x.shape[0]

            global_step, loss_summary, accuracy_summary = sess.run(
                [global_step_op, loss_summary_op, accuracy_summary_op], feed_dict={
                    input: train_x[start_from: to],
                    labels: train_y[start_from: to]
                })
            validation_summary_writer.add_summary(loss_summary, global_step)
            validation_summary_writer.add_summary(accuracy_summary, global_step)

            print(f"Validation accuracy: {accuracy}, loss: {loss}")
            if accuracy > sess.run(val_accuracy_op):
                assign_op = val_accuracy_op.assign(accuracy)
                sess.run(assign_op)
                save_path = val_saver.save(sess, config.ckpt_dev, global_step=global_step)
                print(f"Val Checkpoint saved: {save_path} with accuracy {accuracy}")

    train_summary_writer.close()
    validation_summary_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', dest='t', action='store_true')
    args = parser.parse_args()

    main(args.t)
