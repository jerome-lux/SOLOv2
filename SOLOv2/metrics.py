import tensorflow as tf


class BinaryRecall(tf.keras.metrics.Metric):

    def __init__(self, name='binary_recall', **kwargs):
        super().__init__(name=name, **kwargs)
        self.recall = self.add_weight(name='recall', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, threshold=0.5):
        # Reshape to [nloc, 1]
        y_true, y_pred = tf.reshape(y_true, (-1, 1)), tf.reshape(y_pred, (-1, 1))
        npos = tf.reduce_sum(tf.cast(y_true, tf.float32))
        y_true = tf.cast(y_true, tf.bool)
        y_pred = y_pred > threshold

        tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        tp = tf.reduce_sum(tf.cast(tp, tf.float32))

        self.recall.assign_add(tf.math.divide_no_nan(tp, npos))
        self.total.assign_add(1)

    def result(self):
        return tf.math.divide_no_nan(self.recall, self.total)

    def reset_state(self):
        self.recall.assign(0.)
        self.total.assign(0.)


class BinaryPrecision(tf.keras.metrics.Metric):

    def __init__(self, name='binary_precision', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = self.add_weight(name='precision', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, threshold=0.5):
        # Reshape to [nloc, 1]
        y_true, y_pred = tf.reshape(y_true, (-1, 1)), tf.reshape(y_pred, (-1, 1))
        y_true = tf.cast(y_true, tf.bool)
        y_pred = y_pred > threshold
        npos = tf.reduce_sum(tf.cast(y_pred, tf.float32))

        tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        tp = tf.reduce_sum(tf.cast(tp, tf.float32))

        self.precision.assign_add(tf.math.divide_no_nan(tp, npos))
        self.total.assign_add(1)

    def result(self):
        return tf.math.divide_no_nan(self.precision, self.total)

    def reset_state(self):
        self.precision.assign(0.)
        self.total.assign(0.)

