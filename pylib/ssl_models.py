"""
Consistency regularisation models.
"""

import numpy as np
import tensorflow as tf

from pylib.loss_functions import (
    loss_consistency,
    loss_supervised,
    loss_supervised_masked,
)

# Base -------------------------------------------------------------------------


class BaseModel:
    """
    A supervised-only model.

    We will subclass this model and overwrite the training steps to produce the
    other models.
    """

    def __init__(self, model, optimizer, epochs) -> None:
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs

        self.train_ls = tf.keras.metrics.Mean(name="train_ls")
        self.train_lc = tf.keras.metrics.Mean(name="train_lc")
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy"
        )
        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="test_accuracy"
        )

    def train(self, train_ds, test_ds, batchsize=32, **kwargs):
        """Execute the training loop on the given data."""
        train_batches = train_ds.batch(batchsize)
        test_batches = test_ds.batch(batchsize)

        for epoch in range(self.epochs):
            self._reset_metrics()

            # The third element of each training batch are batch indices.
            # We only need these for the TemporalEnsemble
            for features, labels, _ in train_batches:
                pred, ls, lc, lt = self._train_step(
                    features, labels, epoch=epoch, **kwargs
                )
                self._update_metrics_train(pred, labels, ls, lc, lt)

            for features, labels in test_batches:
                pred, loss = self._test_step(features, labels)
                self._update_metrics_test(pred, labels, loss)

            self._report_metrics(epoch)

    @tf.function
    def _train_step(self, features, labels, **kwargs):
        """Supervised-only mini-batch training step"""
        with tf.GradientTape() as tape:
            logits_student = self.model(features, training=True)

            # Unlabeled obs have 'label' -1
            is_labeled = tf.not_equal(-1, labels)

            loss = tf.cond(
                tf.reduce_any(is_labeled),
                lambda: loss_supervised_masked(labels, logits_student, is_labeled),
                lambda: tf.constant(0, tf.float32),
            )

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return logits_student, loss, -1, loss

    @tf.function
    def _test_step(self, features, labels):
        """One mini-batch test step"""
        logits = self.model(features, training=False)
        loss = loss_supervised(labels, logits)
        return logits, loss

    def _reset_metrics(self):
        self.train_ls.reset_states()
        self.train_lc.reset_states()
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()

    def _update_metrics_train(
        self, pred, labels, loss_supervised, loss_consistency, loss_total
    ):
        self.train_ls(loss_supervised)
        self.train_lc(loss_consistency)
        self.train_loss(loss_total)
        self.train_accuracy(labels, pred)

    def _update_metrics_test(self, pred, labels, loss):
        self.test_loss(loss)
        self.test_accuracy(labels, pred)

    def _report_metrics(self, epoch):
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {self.train_loss.result():.2f} ({self.train_ls.result():.2f}; {self.train_lc.result():.2f}), "
            f"Accuracy: {self.train_accuracy.result() * 100:.2f}, "
            f"Test Loss: {self.test_loss.result():.2f}, "
            f"Test Accuracy: {self.test_accuracy.result() * 100:.2f}"
        )


# Pi model ---------------------------------------------------------------------


class PiModel(BaseModel):
    """The Pi Model"""

    @tf.function
    def _train_step(self, features, labels, **kwargs):
        """Pi Model mini-batch training step"""
        with tf.GradientTape() as tape:
            logits_student = self.model(features, training=True)
            logits_teacher = self.model(features, training=True)
            tf.stop_gradient(logits_teacher)

            # Unlabeled obs have 'label' -1
            is_labeled = tf.not_equal(-1, labels)

            ls = tf.cond(
                tf.reduce_any(is_labeled),
                lambda: loss_supervised_masked(labels, logits_student, is_labeled),
                lambda: tf.constant(0, tf.float32),
            )
            lc = loss_consistency(logits_student, logits_teacher, **kwargs)
            loss = ls + lc

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return logits_student, ls, lc, loss


# Temporal ensemble ------------------------------------------------------------


class TemporalEnsembleModel(BaseModel):
    """The Temporal Ensemble Model"""

    def train(self, train_ds, test_ds, batchsize=32, **kwargs):
        """Execute the training loop on the given data."""
        train_batches = train_ds.batch(batchsize)
        test_batches = test_ds.batch(batchsize)

        # A bit of a hack to initialise the teacher logits
        for features, labels, _ in train_batches:
            pred = self.model(features, training=False)
        logits_teacher = np.zeros((len(train_ds), pred.shape[1]), dtype=np.float32)

        for epoch in range(self.epochs):
            self._reset_metrics()

            for features, labels, idx in train_batches:
                idx = idx.numpy()
                batch_teacher = tf.convert_to_tensor(logits_teacher[idx])

                pred, ls, lc, lt = self._train_step(
                    features,
                    labels,
                    epoch=epoch,
                    logits_teacher=batch_teacher / (1 - kwargs["alpha"] ** (epoch + 1)),
                    **kwargs,
                )

                logits_teacher[idx] = (
                    kwargs["alpha"] * batch_teacher + (1 - kwargs["alpha"]) * pred
                )

                self._update_metrics_train(pred, labels, ls, lc, lt)

            for features, labels in test_batches:
                pred, loss = self._test_step(features, labels)
                self._update_metrics_test(pred, labels, loss)

            self._report_metrics(epoch)

    @tf.function
    def _train_step(self, features, labels, **kwargs):
        """Pi Model mini-batch training step"""
        with tf.GradientTape() as tape:
            logits_student = self.model(features, training=True)

            # Unlabeled obs have 'label' -1
            is_labeled = tf.not_equal(-1, labels)

            ls = tf.cond(
                tf.reduce_any(is_labeled),
                lambda: loss_supervised_masked(labels, logits_student, is_labeled),
                lambda: tf.constant(0, tf.float32),
            )
            lc = loss_consistency(logits_student, **kwargs)

            loss = ls + lc

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return logits_student, ls, lc, loss
