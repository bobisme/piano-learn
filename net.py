from __future__ import print_function
from itertools import izip
import re

import numpy
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn

import tools


class Net(object):
    LABEL_SIZE = 170

    def __init__(self, model_dir):
        self.model_dir = model_dir

    @classmethod
    def build_run_eval(
            cls, model_dir, train_features, train_labels,
            test_features_labels):
        nn = cls(model_dir)
        nn.build_classifier()
        nn.train_and_evaluate(
            train_features, train_labels, test_features_labels, batch=5000)
        return nn

    def _build(self, features, labels, mode):
        FEATURE_SIZE = features.get_shape().as_list[-1]
        LABEL_SIZE = labels.get_shape().as_list[-1]

        # Input layer
        input_layer = tf.reshape(features, [-1, FEATURE_SIZE])

        # Dense Layer 1
        dense = tf.layers.dense(
            inputs=tf.cast(input_layer, tf.float32), units=512,
            activation=tf.nn.relu)
        # Apply dropout regularization to our dense layer.
        # Rate of 0.4 means 40% of elements will be randomly dropped durring
        # training.
        dropout = tf.layers.dropout(
          inputs=dense, rate=0.4, training=(mode == learn.ModeKeys.TRAIN))

        # Logits Layer
        # Defaults to linear activation.
        logits = tf.layers.dense(inputs=dropout, units=LABEL_SIZE)

        final_tensor = tf.nn.sigmoid(logits, name='sigmoided_logits')

        loss = None
        train_op = None

        cast_labels = tf.cast(labels, tf.float32)

        # Calculate Loss (for both TRAIN and EVAL modes)
        if mode != learn.ModeKeys.INFER:
            loss = tf.losses.sigmoid_cross_entropy(cast_labels, logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == learn.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.001,
                optimizer="SGD")

        # Generate Predictions
        predicted_classes = tf.round(final_tensor, name='predicted_classes')
        correct_predictions = tf.equal(predicted_classes, cast_labels)
        all_labels_true = tf.reduce_min(
            tf.cast(correct_predictions, tf.float32), 1)
        accuracy = tf.reduce_mean(all_labels_true, name='mean_min')
        tf.summary.scalar('accuracy', accuracy)

        predictions = {
            'predicted_classes': predicted_classes,
        }

        # Return a ModelFnOps object
        return model_fn.ModelFnOps(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op)

    def get_logging_hook(self):
        # Set up logging for predictions
        tensors_to_log = {
            # "probabilities": "sigmoided_logits",
            'accuracy': 'mean_min',
        }
        return tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=1000)

    def build_classifier(self):
        # Create the Estimator
        self.classifier = learn.Estimator(
            model_fn=self._build, model_dir=self.model_dir)
        return self

    def train(self, features, labels):
        features_, labels_ = self.transform(features, labels)
        self.classifier.fit(
            x=features,
            y=labels,
            batch_size=100,
            steps=10000,
            monitors=[self.get_logging_hook()])

    def get_metrics(self):
        return {
            "accuracy": learn.MetricSpec(
                metric_fn=tf.metrics.accuracy,
                prediction_key="predicted_classes"),
            "precision": learn.MetricSpec(
                metric_fn=tf.metrics.precision,
                prediction_key="predicted_classes"),
            "recall": learn.MetricSpec(
                metric_fn=tf.metrics.recall,
                prediction_key="predicted_classes"),
        }

    def evaluate(self, features_labels, eval_steps=1000):
        for x, y in features_labels:
            def input_fn():
                return (tf.constant(x), tf.constant(y))
            x_, y_ = self.transform(x, y)
            #eval_results = self.classifier.evaluate(
            #    input_fn=input_fn, metrics=self.get_metrics(), steps=eval_steps)
            eval_results = learn.SKCompat(self.classifier).score(
                x, y, metrics=self.get_metrics())
            print(eval_results)
            print('=' * 80)

    def transform(self, features, labels):
        return (features, labels)

    def train_and_evaluate(
            self, features, labels, train_features_labels,
            batch=10000, eval_steps=1000):
        batch_count = int(numpy.ceil(float(features.shape[0]) / batch))
        for offset_number in xrange(batch_count):
            offset = offset_number * batch
            offset_end = offset + batch
            features_slice = features[offset:offset_end]
            label_slice = labels[offset:offset_end]
            print('training chunk {:03d}/{:03d}'.format(
                offset_number + 1, batch_count))
            self.train(features_slice, label_slice)
            print('evaluating...')
            self.evaluate(train_features_labels, eval_steps=eval_steps)

    def transcribe(self, wav_filename):
        frames = tools.wav_to_frames(wav_filename)
        predictions = self.classifier.predict(frames)
        return tools.labels_to_midi(
            (x['predicted_classes'] for x in predictions))


class Net2(Net):
    def _build(self, features, labels, mode):
        FEATURE_SIZE = features.get_shape().as_list[-1]
        LABEL_SIZE = labels.get_shape().as_list[-1]

        is_training = (mode == learn.ModeKeys.TRAIN)

        # Input layer
        input_layer = tf.reshape(features, [-1, FEATURE_SIZE])

        hidden_1 = tf.layers.dense(
            inputs=tf.cast(input_layer, tf.float32),
            units=500, activation=tf.nn.relu)
        dropout_1 = tf.layers.dropout(
            inputs=hidden_1, rate=0.4, training=is_training)
        hidden_2 = tf.layers.dense(
            inputs=dropout_1, units=500, activation=tf.nn.relu)
        dropout_2 = tf.layers.dropout(
            inputs=hidden_2, rate=0.4, training=is_training)
        hidden_3 = tf.layers.dense(
            inputs=dropout_2, units=500, activation=tf.nn.relu)
        dropout_3 = tf.layers.dropout(
            inputs=hidden_3, rate=0.4, training=is_training)

        # Logits Layer
        # Defaults to linear activation.
        logits = tf.layers.dense(inputs=dropout_3, units=LABEL_SIZE)

        final_tensor = tf.nn.sigmoid(logits, name='sigmoided_logits')

        loss = None
        train_op = None

        cast_labels = tf.cast(labels, tf.float32)

        # Calculate Loss (for both TRAIN and EVAL modes)
        if mode != learn.ModeKeys.INFER:
            loss = tf.losses.sigmoid_cross_entropy(cast_labels, logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == learn.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.001,
                optimizer="SGD")

        # Generate Predictions
        predicted_classes = tf.round(final_tensor, name='predicted_classes')
        correct_predictions = tf.equal(predicted_classes, cast_labels)
        all_labels_true = tf.reduce_min(
            tf.cast(correct_predictions, tf.float32), 1)
        accuracy = tf.reduce_mean(all_labels_true, name='mean_min')
        tf.summary.scalar('accuracy', accuracy)
        # tf.summary.scalar('all_labels_true', all_labels_true)

        predictions = {
            'predicted_classes': predicted_classes,
        }

        # Return a ModelFnOps object
        return model_fn.ModelFnOps(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op)
# nn2 = Net2('/tmp/nn2-2')
# nn2.build_classifier()
# nn2.train_and_evaluate(X_train, y_train, [(X_test, y_test)], batch=50000)

class NetRNN(Net):
    def get_blocks(self, tensor, frame_count, stride=1):
        blocks = []
        total_frames, feature_size = tensor.shape
        lo_range = xrange(0, total_frames, stride)
        hi_range = xrange(frame_count, total_frames + 1, stride)
        for lo, hi in izip(lo_range, hi_range):
            blocks.append(tensor[lo:hi])
        return numpy.array(blocks)

    def transform(self, features, labels):
        print('converting to windows')
        feats = self.get_blocks(features, 4)
        print('did that')
        return feats, labels

    def _build(self, features, labels, mode):
        FEATURE_COUNT, FEATURE_SIZE = all_features.shape
        LABEL_SIZE = labels.get_shape().as_list()[-1]

        is_training = (mode == learn.ModeKeys.TRAIN)

        # Input layer
        frames_per_window = 4
        print(features.shape)
        input_layer = tf.cast(tf.reshape(
            features, [-1, frames_per_window, FEATURE_SIZE]), tf.float64)
        sequences = tf.unstack(input_layer, frames_per_window, 1)

        rnn_cell = tf.contrib.rnn.BasicLSTMCell(400)
        rnn_out, rnn_state = tf.contrib.rnn.static_rnn(
            rnn_cell, sequences, dtype=tf.float64)
        #for x in rnn_out:
        #    print(x)
        # Logits Layer
        # Defaults to linear activation.
        logits = tf.layers.dense(
            inputs=tf.concat(rnn_out, 0), units=LABEL_SIZE)

        final_tensor = tf.nn.sigmoid(logits, name='sigmoided_logits')

        loss = None
        train_op = None

        cast_labels = tf.cast(labels, tf.float64)

        # Calculate Loss (for both TRAIN and EVAL modes)
        if mode != learn.ModeKeys.INFER:
            loss = tf.losses.sigmoid_cross_entropy(cast_labels, logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == learn.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.001,
                optimizer="SGD")

        # Generate Predictions
        predicted_classes = tf.round(final_tensor, name='predicted_classes')
        correct_predictions = tf.equal(predicted_classes, cast_labels)
        all_labels_true = tf.reduce_min(
            tf.cast(correct_predictions, tf.float32), 1)
        accuracy = tf.reduce_mean(all_labels_true, name='mean_min')
        predicted_class_count = tf.reduce_sum(
            predicted_classes, name='predicted_class_count')
        correct_class_count = tf.reduce_sum(
            tf.cast(correct_predictions, tf.int64), name='correct_class_count')

        with tf.name_scope('summaries'):
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('predicted class count', predicted_class_count)
            tf.summary.scalar('correct class count', correct_class_count)
        # tf.summary.scalar('all_labels_true', all_labels_true)

        predictions = {
            'predicted_classes': predicted_classes,
        }

        # Return a ModelFnOps object
        return model_fn.ModelFnOps(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op)

    def get_logging_hook(self):
        # Set up logging for predictions
        tensors_to_log = {
            "probabilities": "sigmoided_logits",
            "classes": "predicted_classes",
            'accuracy': 'mean_min',
            'predicted_class_count': 'predicted_class_count',
            'correct_class_count': 'correct_class_count',
        }
        return tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=1000)
# rnn1 = NetRNN('/tmp/rnn1-4')
# rnn1.build_classifier()
# rnn1.train_and_evaluate(
#     all_features[:-5000], all_labels[:-5000],
#     [(all_features[-5000:], all_labels[-5000:])], batch=5000)

class NetCNN1(Net):
    def _build(self, features, labels, mode):
        # FEATURE_COUNT, FEATURE_SIZE = all_features.shape
        # LABEL_SIZE = all_labels.shape[1]
        is_training = (mode == learn.ModeKeys.TRAIN)
        label_size = labels.get_shape().as_list()[-1]

        # Input layer
        frames_per_window = 8
        input_layer = tools.unpack_complex(features)

        feature_size = input_layer.get_shape().as_list()[-1]

        input_layer = tf.cast(input_layer, tf.float32)
        input_layer = tools.get_windows(input_layer, frames_per_window)
        input_layer = tf.reshape(
            input_layer, [-1, frames_per_window, feature_size, 1])
        # input_layer = tf.cast(input_layer, tf.float32)

        # Convolutional Layer
        nn = tf.layers.conv2d(input_layer, 100, [5, 5], padding='same')
        nn = tf.layers.max_pooling2d(nn, [2, 2], [1, 1])
        # nn = tf.layers.conv2d(nn, 20, [5, 5], padding='same')
        # nn = tf.layers.max_pooling2d(nn, [2, 2], [1, 1])

        p2_shape = nn.get_shape().as_list()
        # print(p2_shape)
        nn = tf.reshape(
            nn, [-1, p2_shape[1] * p2_shape[2] * p2_shape[3]])
        # flattened = tf.contrib.layers.flatten(conv_2)
        nn = tf.layers.dense(nn, units=512, activation=tf.nn.relu)
        # nn = tf.layers.dropout(nn, rate=0.4, training=is_training)
        logits = tf.layers.dense(nn, units=label_size)

        final_tensor = tf.nn.sigmoid(logits, name='sigmoided_logits')

        loss = None
        train_op = None

        cast_labels = tf.cast(labels, tf.float32)

        # Calculate Loss (for both TRAIN and EVAL modes)
        if mode != learn.ModeKeys.INFER:
            loss = tf.losses.sigmoid_cross_entropy(cast_labels, logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == learn.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.001,
                optimizer="SGD")

        # Generate Predictions
        predicted_classes = tf.round(final_tensor, name='predicted_classes')
        correct_predictions = tf.equal(predicted_classes, cast_labels)
        all_labels_true = tf.reduce_min(
            tf.cast(correct_predictions, tf.float32), 1)
        accuracy = tf.reduce_mean(all_labels_true, name='mean_min')
        predicted_class_count = tf.reduce_sum(
            predicted_classes, name='predicted_class_count')
        correct_class_count = tf.reduce_sum(
            tf.cast(correct_predictions, tf.int64), name='correct_class_count')

        with tf.name_scope('summaries'):
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('predicted class count', predicted_class_count)
            tf.summary.scalar('correct class count', correct_class_count)
        # tf.summary.scalar('all_labels_true', all_labels_true)

        predictions = {
            'predicted_classes': predicted_classes,
        }

        # Return a ModelFnOps object
        return model_fn.ModelFnOps(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op)

    def get_logging_hook(self):
        # Set up logging for predictions
        tensors_to_log = {
            "probabilities": "sigmoided_logits",
            "classes": "predicted_classes",
            'accuracy': 'mean_min',
            'predicted_class_count': 'predicted_class_count',
            'correct_class_count': 'correct_class_count',
        }
        return tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=1000)
# cnn1 = NetCNN1('/tmp/cnn1-2')
# cnn1.build_classifier()
# cnn1.train_and_evaluate(
#     all_features[:-10000], all_labels[:-10000],
#     [(all_features[-10000:], all_labels[-10000:])], batch=5000)

class NetNN2(NetCNN1):
    def _build(self, features, labels, mode):
        is_training = (mode == learn.ModeKeys.TRAIN)

        # Input layer
        frames_per_window = 8
        nn = tools.unpack_complex(features)

        feature_size = nn.get_shape().as_list()[-1]

        nn = tf.cast(nn, tf.float32)
        # nn = tools.get_windows(nn, frames_per_window)
        # input_layer = tf.reshape(
        #     input_layer, [-1, frames_per_window, feature_size, 1])
        # input_layer = tf.cast(input_layer, tf.float32)

        # Convolutional Layer
        # nn = tf.layers.conv2d(nn, 100, [5, 5], padding='same')
        # nn = tf.layers.max_pooling2d(nn, [2, 2], [1, 1])
        # nn = tf.layers.conv2d(nn, 20, [5, 5], padding='same')
        # nn = tf.layers.max_pooling2d(nn, [2, 2], [1, 1])

        # p2_shape = nn.get_shape().as_list()
        # # print(p2_shape)
        # nn = tf.reshape(
        #     nn, [-1, p2_shape[1] * p2_shape[2] * p2_shape[3]])
        # flattened = tf.contrib.layers.flatten(conv_2)
        nn = tf.layers.dense(nn, units=512, activation=tf.nn.relu)
        # nn = tf.layers.dropout(nn, rate=0.4, training=is_training)
        logits = tf.layers.dense(nn, units=self.LABEL_SIZE)

        final_tensor = tf.nn.sigmoid(logits, name='sigmoided_logits')

        loss = None
        train_op = None

        if labels:
            cast_labels = tf.cast(labels, tf.float32)

        # Calculate Loss (for both TRAIN and EVAL modes)
        if mode != learn.ModeKeys.INFER:
            loss = tf.losses.sigmoid_cross_entropy(cast_labels, logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == learn.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.001,
                optimizer="SGD")

        # Generate Predictions
        predicted_classes = tf.round(final_tensor, name='predicted_classes')

        if labels:
            correct_predictions = tf.equal(predicted_classes, cast_labels)
            all_labels_true = tf.reduce_min(
                tf.cast(correct_predictions, tf.float32), 1)
            accuracy = tf.reduce_mean(all_labels_true, name='mean_min')
            predicted_class_count = tf.reduce_sum(
                predicted_classes, name='predicted_class_count')
            correct_class_count = tf.reduce_sum(
                tf.cast(correct_predictions, tf.int64),
                name='correct_class_count')

            with tf.name_scope('summaries'):
                tf.summary.scalar('accuracy', accuracy)
                tf.summary.scalar(
                    'predicted class count', predicted_class_count)
                tf.summary.scalar('correct class count', correct_class_count)
        # tf.summary.scalar('all_labels_true', all_labels_true)

        predictions = {
            'predicted_classes': predicted_classes,
        }

        # Return a ModelFnOps object
        return model_fn.ModelFnOps(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op)

    def get_logging_hook(self):
        # Set up logging for predictions
        tensors_to_log = {
            "probabilities": "sigmoided_logits",
            "classes": "predicted_classes",
            'accuracy': 'mean_min',
            'predicted_class_count': 'predicted_class_count',
            'correct_class_count': 'correct_class_count',
        }
        return tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=1000)
# cnn1 = NetCNN1('/tmp/cnn1-2')
# cnn1.build_classifier()
# cnn1.train_and_evaluate(
#     all_features[:-10000], all_labels[:-10000],
#     [(all_features[-10000:], all_labels[-10000:])], batch=5000)

class NetNN4(NetCNN1):
    def _build(self, features, labels, mode):
        is_training = (mode == learn.ModeKeys.TRAIN)

        # Input layer
        frames_per_window = 4
        nn = tools.unpack_complex(features)

        feature_size = nn.get_shape().as_list()[-1]

        nn = tf.cast(nn, tf.float32)
        nn = tools.get_windows(nn, frames_per_window)
        # nn = tf.reshape(nn, [-1, frames_per_window * feature_size])
        # nn = tf.reshape(nn, [-1, frames_per_window, feature_size, 1])

        # Convolutional Layer
        # nn = tf.layers.conv2d(nn, 100, [5, 5], padding='same')
        # nn = tf.layers.max_pooling2d(nn, [2, 2], [1, 1])
        # nn = tf.layers.conv2d(nn, 20, [5, 5], padding='same')
        # nn = tf.layers.max_pooling2d(nn, [2, 2], [1, 1])

        # shape = nn.get_shape().as_list()
        # nn = tf.reshape(nn, [-1, shape[1] * shape[2] * shape[3]])
        # flattened = tf.contrib.layers.flatten(conv_2)
        nn = tf.layers.dense(nn, units=5000, activation=tf.nn.relu)
        # nn = tf.layers.dense(nn, units=200, activation=tf.nn.relu)
        # nn = tf.layers.dense(nn, units=100, activation=tf.nn.relu)
        # nn = tf.layers.dropout(nn, rate=0.4, training=is_training)
        logits = tf.layers.dense(nn, units=self.LABEL_SIZE)

        final_tensor = tf.nn.sigmoid(logits, name='sigmoided_logits')

        loss = None
        train_op = None

        if labels is not None:
            cast_labels = tf.cast(labels, tf.float32)

        # Calculate Loss (for both TRAIN and EVAL modes)
        if mode != learn.ModeKeys.INFER:
            loss = tf.losses.sigmoid_cross_entropy(cast_labels, logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == learn.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.001,
                optimizer="SGD")

        # Generate Predictions
        predicted_classes = tf.round(final_tensor, name='predicted_classes')

        if labels is not None:
            correct_predictions = tf.equal(predicted_classes, cast_labels)
            all_labels_true = tf.reduce_min(
                tf.cast(correct_predictions, tf.float32), 1)
            accuracy = tf.reduce_mean(all_labels_true, name='mean_min')
            predicted_class_count = tf.reduce_sum(
                predicted_classes, name='predicted_class_count')
            correct_class_count = tf.reduce_sum(
                tf.cast(correct_predictions, tf.int64),
                name='correct_class_count')

            with tf.name_scope('summaries'):
                tf.summary.scalar('accuracy', accuracy)
                tf.summary.scalar(
                    'predicted class count', predicted_class_count)
                tf.summary.scalar('correct class count', correct_class_count)
        # tf.summary.scalar('all_labels_true', all_labels_true)

        predictions = {
            'predicted_classes': predicted_classes,
        }

        # Return a ModelFnOps object
        return model_fn.ModelFnOps(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op)

    def get_logging_hook(self):
        # Set up logging for predictions
        tensors_to_log = {
            "probabilities": "sigmoided_logits",
            "classes": "predicted_classes",
            'accuracy': 'mean_min',
            'predicted_class_count': 'predicted_class_count',
            'correct_class_count': 'correct_class_count',
        }
        return tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=1000)


class CNN2(NetCNN1):
    def _build(self, features, labels, mode):
        is_training = (mode == learn.ModeKeys.TRAIN)

        # Input layer
        frames_per_window = 4
        nn = tools.unpack_complex(features)

        feature_size = nn.get_shape().as_list()[-1]

        nn = tf.cast(nn, tf.float32)
        nn = tools.get_windows(nn, frames_per_window)
        nn = tf.reshape(nn, [-1, frames_per_window, feature_size, 1])

        # Convolutional Layer
        nn = tf.layers.conv2d(nn, 200, [4, 4], padding='same')
        nn = tf.layers.max_pooling2d(nn, [2, 2], [1, 1])
        nn = tf.layers.conv2d(nn, 200, [3, 3], padding='same')
        nn = tf.layers.max_pooling2d(nn, [2, 2], [1, 1])

        shape = nn.get_shape().as_list()
        nn = tf.reshape(nn, [-1, shape[1] * shape[2] * shape[3]])
        # flattened = tf.contrib.layers.flatten(conv_2)
        nn = tf.layers.dense(nn, units=4000, activation=tf.nn.sigmoid)
        # nn = tf.layers.dense(nn, units=200, activation=tf.nn.relu)
        # nn = tf.layers.dense(nn, units=100, activation=tf.nn.relu)
        # nn = tf.layers.dropout(nn, rate=0.4, training=is_training)
        logits = tf.layers.dense(nn, units=self.LABEL_SIZE)

        final_tensor = tf.nn.sigmoid(logits, name='sigmoided_logits')

        loss = None
        train_op = None

        if labels is not None:
            cast_labels = tf.cast(labels, tf.float32)

        # Calculate Loss (for both TRAIN and EVAL modes)
        if mode != learn.ModeKeys.INFER:
            loss = tf.losses.sigmoid_cross_entropy(cast_labels, logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == learn.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.001,
                optimizer="SGD")

        # Generate Predictions
        predicted_classes = tf.round(final_tensor, name='predicted_classes')

        if labels is not None:
            correct_predictions = tf.equal(predicted_classes, cast_labels)
            all_labels_true = tf.reduce_min(
                tf.cast(correct_predictions, tf.float32), 1)
            accuracy = tf.reduce_mean(all_labels_true, name='mean_min')
            predicted_class_count = tf.reduce_sum(
                predicted_classes, name='predicted_class_count')
            correct_class_count = tf.reduce_sum(
                tf.cast(correct_predictions, tf.int64),
                name='correct_class_count')

            with tf.name_scope('summaries'):
                tf.summary.scalar('accuracy', accuracy)
                tf.summary.scalar(
                    'predicted class count', predicted_class_count)
                tf.summary.scalar('correct class count', correct_class_count)
        # tf.summary.scalar('all_labels_true', all_labels_true)

        predictions = {
            'predicted_classes': predicted_classes,
        }

        # Return a ModelFnOps object
        return model_fn.ModelFnOps(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op)

    def get_logging_hook(self):
        # Set up logging for predictions
        tensors_to_log = {
            "probabilities": "sigmoided_logits",
            "classes": "predicted_classes",
            'accuracy': 'mean_min',
            'predicted_class_count': 'predicted_class_count',
            'correct_class_count': 'correct_class_count',
        }
        return tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=1000)


class SimpleNN(NetCNN1):
    def _build(self, features, labels, mode):
        is_training = (mode == learn.ModeKeys.TRAIN)

        # Input layer
        nn = tools.unpack_complex(features)

        feature_size = nn.get_shape().as_list()[-1]

        nn = tf.cast(nn, tf.float32)
        nn = tf.layers.dense(nn, units=512, activation=tf.nn.relu)
        # nn = tf.layers.dropout(nn, rate=0.4, training=is_training)
        logits = tf.layers.dense(nn, units=self.LABEL_SIZE)

        final_tensor = tf.nn.sigmoid(logits, name='sigmoided_logits')

        loss = None
        train_op = None

        if labels is not None:
            cast_labels = tf.cast(labels, tf.float32)

        # Calculate Loss (for both TRAIN and EVAL modes)
        if mode != learn.ModeKeys.INFER:
            loss = tf.losses.sigmoid_cross_entropy(cast_labels, logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == learn.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.001,
                optimizer="SGD")

        # Generate Predictions
        predicted_classes = tf.round(final_tensor, name='predicted_classes')

        if labels is not None:
            correct_predictions = tf.equal(predicted_classes, cast_labels)
            all_labels_true = tf.reduce_min(
                tf.cast(correct_predictions, tf.float32), 1)
            accuracy = tf.reduce_mean(all_labels_true, name='mean_min')
            predicted_class_count = tf.reduce_sum(
                predicted_classes, name='predicted_class_count')
            correct_class_count = tf.reduce_sum(
                tf.cast(correct_predictions, tf.int64),
                name='correct_class_count')

            with tf.name_scope('summaries'):
                tf.summary.scalar('accuracy', accuracy)
                tf.summary.scalar(
                    'predicted class count', predicted_class_count)
                tf.summary.scalar('correct class count', correct_class_count)
        # tf.summary.scalar('all_labels_true', all_labels_true)

        predictions = {
            'predicted_classes': predicted_classes,
        }

        # Return a ModelFnOps object
        return model_fn.ModelFnOps(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op)
