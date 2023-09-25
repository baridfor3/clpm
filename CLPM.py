# -*- coding: utf-8 -*-
# code warrior: Barid
import numpy as np
import tensorflow as tf
from UNIVERSAL.basic_layer import embedding_layer, layerNormalization_layer
from UNIVERSAL.utils import cka, staticEmbedding_util
from UNIVERSAL.basic_metric import mean_metric

_NEG_INF_FP32 = -1e9
_NEG_INF_FP16 = np.finfo(np.float16).min


def input_preprocessing(x, non_padding=None):
    """
    apply non_paddings to x.
    non_padding = [1,1,1,1,0,0,0]
    x  = x * non_paddin

    """
    if non_padding is not None:
        x *= non_padding
    return x


def input_preprocess(inputs, position_index=None, pre_fn=None, **kwargs):
    # obseving the model could not understand the distinguish Between
    # step position and position encoding becasu they have the same value.
    if "max_seq" in kwargs:
        max_seq = kwargs["max_seq"]
    else:
        max_seq = 1000
    if position_index is not None:
        length = max_seq
    else:
        length = None
    if pre_fn is None:
        inputs = staticEmbedding_util.add_position_timing_signal(
            inputs, 0, position=position_index, length=length
        )
    else:
        inputs += pre_fn(inputs)
    return inputs


class CLPM_EmbeddingSharedWeights(embedding_layer.EmbeddingSharedWeights):
    """Calculates input embeddings and pre-softmax linear with shared weights."""

    def __init__(
        self,
        vocab_size,
        num_units,
        pad_id=0,
        mask_token=4,
        scale_we=True,
        affine=False,
        name="embedding",
        domain_index=[],
    ):
        """Specify characteristic parameters of embedding layer.
        Args:
          vocab_size: Number of tokens in the embedding.
          num_units: Dimensionality of the embedding.
          pad_id: Default 0.
          mask_id: Default 0.
          affine:Default True.
          domain_index: The domain list [[x,x,x,x],[y,y,y,y]].
                    NOTE that each domain should include common tokens like [EOS], [PADDING], [SOS], etc.
        """
        super(CLPM_EmbeddingSharedWeights, self).__init__(
            vocab_size=vocab_size,
            num_units=num_units,
            pad_id=pad_id,
            affine=affine,
            scale_we=scale_we,
            domain_index=domain_index,
            name=name,
        )
        # super().__init__(param, **kwargs)
        self.mask_token = mask_token
        self.domain_flag = 0
        self.num_heads = 8
        self.domain_index = domain_index
        if len(self.domain_index) > 0:
            self.set_domain_bias(self.domain_index)
            self.rebuild_domain_index(self.domain_index)
        self.discriminator_LN = layerNormalization_layer.LayerNorm()
        self.discriminator_1 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu)
        self.discriminator_2 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu)
        self.discriminator_3 = tf.keras.layers.Dense(1)
        self.en_proto = mean_metric.Mean_MetricLayer("En_prototype")
        self.de_proto = mean_metric.Mean_MetricLayer("De_prototype")
        # self.de_analogy = tf.keras.metrics.Mean("De_analogy")
        # self.en_analogy = tf.keras.metrics.Mean("En_analogy")
        self.de_domain = mean_metric.Mean_MetricLayer("De")
        self.en_domain = mean_metric.Mean_MetricLayer("En")

    def discriminator(self, x, training=True):
        x = tf.stop_gradient(x)
        x = self.discriminator_LN(x)
        x = self.discriminator_1(x)
        if training:
            x = tf.nn.dropout(x, rate=0.1)
        x = self.discriminator_2(x)
        if training:
            x = tf.nn.dropout(x, rate=0.1)
        x = self.discriminator_3(x)
        return tf.nn.sigmoid(x)

    def set_domain_bias(self, domain):
        self.domain_bias_matrix = tf.zeros([1, self.vocab_size])
        for d in domain:
            dom = self.domain_filter(self.vocab_size, d)
            self.domain_bias_matrix = tf.concat(
                (self.domain_bias_matrix, tf.reshape(dom, [1, self.vocab_size])), 0
            )

    def rebuild_domain_index(self, domain):
        self.r_domain_index = tf.zeros([1, self.vocab_size])
        for domain_index in domain:
            updates = tf.ones(len(domain_index))
            re = tf.scatter_nd(domain_index, updates, tf.constant([self.vocab_size]))
            self.r_domain_index = tf.concat(
                (self.r_domain_index, tf.reshape(re, [1, self.vocab_size])), 0
            )

    def _CLPM(
        self,
        clpm_position,
        label_x,
        src,
        cross_lingual_id,
        K=3,
        model_fn=lambda x: x,
        lang_fn=lambda x: x,
    ):
        """
        masked_x = x,[mask_id], x,[clpm_id], x (any masking strategy)
        clpm_position = 0,0,0,1,0 (whether a position requires CLPM)
        label_x = x,x,x,x (the original input)
        NOTE:
        we compute every CLPM in paralell.
        """

        def _cos(Q, k=3, self_bias=0):
            domain_bias = tf.gather(self.domain_bias_matrix, cross_lingual_id)
            # domain bias blocks out "not in the language domain" token by setting. Just like we use the masking in transformer.
            w, p = tf.nn.top_k(Q + domain_bias + self_bias, k=k)
            return w, p
        guess_cross = src
        x = tf.stop_gradient(self._embedding(x))
        guess_cross = tf.stop_gradient(self._embedding(guess_cross))
        cross_lang = tf.stop_gradient(lang_fn(cross_lingual_id))
        # get last hiden state
        guess_cross = model_fn(guess_cross + cross_lang)
        # get output probability over the vocabulary
        Q = self._linear(guess_cross)
        # select top-k
        w, p = _cos(Q, 3)
        p = p * tf.expand_dims(clpm_position, -1)
        x = tf.stop_gradient(x)
        p = self._embedding(p)
        # compute weigts
        weights = tf.matmul(tf.expand_dims(x, -2),p,transpose_b=True,)
        weights = tf.nn.softmax(weights)
        # obtain final [C]
        clpm = tf.matmul(
            tf.stop_gradient(weights),
            tf.stop_gradient(p),
        )

        clpm_plus = tf.squeeze(clpm, -2) * tf.cast(tf.expand_dims(clpm_position, -1), tf.float32)
        return clpm_plus, 0

    def recognizer(self, x, shift, mask, cross_lingual_id):
        # 1 de, 2 en
        en_marker = tf.cast(tf.equal(cross_lingual_id, 2), tf.float32)
        de_marker = tf.cast(tf.equal(cross_lingual_id, 1), tf.float32)
        mask = tf.cast(mask, tf.float32)
        metric = tf.squeeze(self.discriminator(x, training=False), -1) * mask
        self.en_proto(
            tf.math.divide_no_nan(
                tf.reduce_sum(metric * en_marker), tf.reduce_sum(mask * en_marker)
            )
        )
        self.de_proto(
            tf.math.divide_no_nan(
                tf.reduce_sum(metric * de_marker), tf.reduce_sum(mask * de_marker)
            )
        )

    def domain_filter(src, vocabulary, domain_index):
        """
        return [vocabulary]
        """
        updates = tf.ones(len(domain_index))
        re = tf.scatter_nd(domain_index, updates, tf.constant([vocabulary]))
        neg_inf = _NEG_INF_FP16 if re.dtype == tf.float16 else _NEG_INF_FP32
        return (1 - re) * neg_inf

    def get_config(self):
        # config = super(EmbeddingSharedWeights, self).get_config()
        c = {
            "vocab_size": self.vocab_size,
            "num_units": self.num_units,
            "pad_id": self.pad_id,
            "name": self.name,
        }
        # config.update(c)
        return c

    def vis_domain(self, src, n):
        domain_index = self.r_domain_index[2].numpy()
        non_zero = tf.reduce_sum(
            tf.cast(tf.not_equal(tf.reduce_sum(src, -1, keepdims=True), 0.0), tf.float32)
        )
        src = self._linear(src)
        src = tf.squeeze(tf.reduce_sum(src, -2) / non_zero, 0)
        src = src.numpy()
        domain_index = domain_index
        with open("./" + n, "w") as f:
            for i in range(len(src)):
                f.write(str(src[i]) + "@@" + str(domain_index[i]))
                f.write("\n")
