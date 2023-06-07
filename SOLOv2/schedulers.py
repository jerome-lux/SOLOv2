import tensorflow as tf
import math as m
pi = tf.constant(m.pi)


class cosine_decay_scheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr=1e-3, alpha=1e-3, maxsteps=1000):
        super(cosine_decay_scheduler, self).__init__()

        self.initial_lr = initial_lr
        self.alpha = alpha
        self.maxsteps = maxsteps

    def __call__(self, step):
        x = tf.math.minimum(step, self.maxsteps) / self.maxsteps
        cosine_decay = 0.5 * (1 + tf.math.cos(pi * x))
        return ((1 - self.alpha) * cosine_decay + self.alpha) * self.initial_lr
