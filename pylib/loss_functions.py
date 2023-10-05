import tensorflow as tf


def loss_supervised_masked(labels, logits, labeled_mask):
    """Supervised loss"""
    return loss_supervised(
        tf.boolean_mask(labels, labeled_mask), tf.boolean_mask(logits, labeled_mask)
    )


def loss_supervised(labels, logits):
    """Supervised loss"""
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels,
        logits,
        from_logits=True,
    )
    return tf.reduce_mean(loss)


def loss_consistency(logits_student, logits_teacher, **kwargs):
    """Consistency loss"""
    loss = tf.keras.losses.mean_squared_error(
        tf.keras.activations.softmax(logits_student),
        tf.keras.activations.softmax(logits_teacher),
    )
    # loss = tf.keras.losses.kl_divergence(logits_student, logits_teacher)

    weight = kwargs["gamma"] * consistency_weight(kwargs["epoch"], kwargs["tau"])

    return weight * tf.reduce_mean(loss)


def consistency_weight(epoch, tau):
    """The Gaussian ramp-up curve, exp[-5(1 - T)^2], used by Laine and Aila"""
    if epoch >= tau:
        return tf.constant(1.0, tf.float32)

    warmup = tf.cast(1 - epoch / tau, tf.float32)
    return tf.exp(tf.constant(-5, tf.float32) * tf.square(warmup))
