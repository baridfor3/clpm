# -*- coding: utf-8 -*-
# code warrior: Barid
import sys
from functools import partial

import tensorflow as tf
from UNIVERSAL.MLM import MLM_base
from UNIVERSAL.utils import padding_util

# from UNIVERSAL.block import TransformerBlock
import CLPM
import initialization as init


class CLPM_MLM(MLM_base.MLM_base):
    def __init__(self, param, **kwargs):
        super().__init__(param, **kwargs)
        self.embedding_softmax_layer = CLPM.CLPM_EmbeddingSharedWeights(
            param["vocabulary_size"],
            param["num_units"],
            domain_index=self.param["domain_index"],
            mask_token=param["MASK_ID"],
            affine=param["affine_we"],
            scale_we=param["scale_we"],
            name="lang_enc" + "_" + str(param["scale_we"]),
        )
        self.probability_generator = self.embedding_softmax_layer._linear

    def pre_training(self, data):
        ((input_src, output_tgt, span, tgt_label, lang_ids),) = data
        src_lang_ids = lang_ids
        tgt_lang_ids = tf.where(tf.equal(lang_ids,1),tf.random.uniform(
            [], minval=2, maxval=self.param["number_of_language"] + 1, dtype=tf.dtypes.int32, seed=None, name=None
        ), tf.constant(1))
        metric = tgt_label
        if 1 in self.param["app"] and 2 in self.param["app"] and len(self.param["app"]) == 2:
            metric = tf.concat([metric, metric], 0)
            tgt_label = tf.concat([tgt_label, tgt_label], 0)
        _ = self.seq2seq_training(
            self.call,
            input_src,
            output_tgt,
            sos=self.param["EOS_ID"],
            src_id=src_lang_ids,
            tgt_id=tgt_lang_ids,
            tgt_label=tgt_label,
            tgt_metric=metric,
            span=span,
        )

    def forward(
        self,
        src,
        tgt,
        training=True,
        attention_bias=0,
        decoder_self_attention_bias=0,
        cache=None,
        encoder_padding=None,
        decoder_padding=None,
        enc_position=None,
        dec_position=None,
        vis=False,
        src_id=None,
        tgt_id=None,
        # clpm_position=None,
        tgt_label=None,
    ):
        """
            src: input sentences, e.g., 1,2,3,4
            tgt: only used for encoder-decoder model, e.g., [start], 1,2,3,4
            src_id: language id of input sentence. no need for XLM-R
            tgt_id: language id of output sentence.
        """
        def _run_encoder(
            src,
            src_id,
            attention_bias,
            training,
            encoder_padding,
            enc_position,
            vis,
        ):

            src_id = tf.ones_like(src) * src_id  # language id of input sentence
            cross_id = tf.ones_like(src) * tgt_id # targe language id.

            if tgt_label != None:
                clpm_position = (
                    tf.cast(tf.not_equal(tgt_label, self.param["EOS_ID"]), tf.int32)
                    * tf.cast(tf.not_equal(tgt_label, self.param["UNK_ID"]), tf.int32)
                    * tf.cast(tf.equal(src, self.param["MASK_ID"]), tf.int32)
                    * tf.cast(tf.not_equal(tgt_label, 0), tf.int32)
                    * tf.cast(
                        tf.keras.backend.random_bernoulli(
                            tf.shape(src), tf.fill(tf.shape(src), self.param["alternation_ratio"]) # 0.5 can be finetune to change the probability of [C]
                        ),
                        tf.int32,
                    )
                )

                _, l = tf.unstack(tf.shape(clpm_position))
                #  Warpping inference mode.
                enc_fn = partial(
                    self.encoding,
                    attention_bias=attention_bias,
                    training=False,
                    # encoder_padding=tf.cast(1 - clpm_position, tf.float32),
                    encoder_padding=encoder_padding,
                    enc_position=enc_position,
                )
                clpm_plus, clpm_minus = self.embedding_softmax_layer._CLPM(
                    clpm_position=clpm_position,
                    src=src,
                    cross_lingual_id=cross_id,
                    model_fn=enc_fn,
                    lang_fn=self.lang_encoding,
                )
                clpm_plus = tf.stop_gradient(clpm_plus)

                clpm_check = tf.cast(
                    tf.greater(
                        tf.cast(self.optimizer.iterations.read_value(), tf.float32), float(self.param["CLPM_warmup"]) # dont use [C] in the early 50000.0 steps
                    ),
                    tf.int32,
                )
                clpm_position = clpm_position * clpm_check
                clpm_position_em = tf.cast(tf.expand_dims(clpm_position,-1),tf.float32)
                src = src * ((1.0 - clpm_position_em)) + tf.stop_gradient(src*clpm_position_em)
                src = src + clpm_plus
                src = tf.where(tf.cast(clpm_position_em,tf.bool),clpm_plus,src)
            else:
                src = self.embedding_softmax_layer(src)
                clpm_position = 0
            if src_id != None:
                src_id = (
                    src_id * (1 - clpm_position) + cross_id * clpm_position
                )
                src_id = self.lang_encoding(src_id)
                src += src_id
            enc = self.encoding(
                src,
                attention_bias,
                # attention_bias+(1-central_masking)*tf.expand_dims(tf.cast(clpm_position,tf.float32),-1),
                training=training,
                encoder_padding=encoder_padding,
                enc_position=enc_position,
                vis=vis,
            )
            return enc
        def _run_decoder(
            tgt,
            tgt_id,
            encATT,
            decoder_self_attention_bias,
            attention_bias,
            training,
            cache,
            dec_position,
            vis,
            decoder_padding,
        ):
            tgt = self.embedding_softmax_layer(tgt)
            if tgt_id != None:
                tgt_id = self.lang_encoding(tgt_id)
                tgt += tgt_id
            dec = self.decoding(
                tgt,
                encATT,
                decoder_self_attention_bias,
                attention_bias,
                training=training,
                cache=cache,
                dec_position=dec_position,
                vis=vis,
                decoder_padding=decoder_padding,
            )
            return dec

        run_encoder = partial(
            _run_encoder,
            attention_bias=attention_bias,
            training=training,
            encoder_padding=encoder_padding,
            enc_position=enc_position,
            vis=vis,
        )
        run_decoder = partial(
            _run_decoder,
            decoder_self_attention_bias=decoder_self_attention_bias,
            attention_bias=attention_bias,
            training=training,
            cache=cache,
            dec_position=dec_position,
            vis=vis,
            decoder_padding=decoder_padding,
        )
        if 1 in self.param["app"] and len(self.param["app"]) == 1:
            logits = run_encoder(src, src_id)
        if 2 in self.param["app"] and len(self.param["app"]) == 1:
            logits = run_decoder(tgt, tgt_id, None)
        if 3 in self.param["app"] and len(self.param["app"]) == 1:
            enc = run_encoder(src, src_id)
            logits = run_decoder(tgt, tgt_id, enc)
        if 1 in self.param["app"] and 2 in self.param["app"] and len(self.param["app"]) == 2:
            enc = run_encoder(src, src_id,)
            dec = run_decoder(tgt, tgt_id, None)
            logits = tf.concat([enc, dec], 0)
        return logits
