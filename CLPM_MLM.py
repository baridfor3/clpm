# -*- coding: utf-8 -*-
# code warrior: Barid
import sys
from functools import partial

import tensorflow as tf
from UNIVERSAL.model import MLM_base
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
            domain_index=init.get_domain_index(),
            mask_token=param["MASK_ID"],
            affine=param["affine_we"],
            scale_we=param["scale_we"],
            name="lang_enc" + "_" + str(param["scale_we"]),
        )
        self.probability_generator = self.embedding_softmax_layer._linear
        self.de_dict_size = len(self.embedding_softmax_layer.domain_index[0])
        self.en_dict_size = len(self.embedding_softmax_layer.domain_index[1])
        # self.encoder = CLPM.TransformerEncoder_CBOW(param)

    def pre_training(self, data):
        ((input_src, output_tgt, span, tgt_label, lang_ids),) = data
        src_lang_ids = tf.reshape(tf.gather(lang_ids, 0, axis=1), [-1, 1])
        tgt_lang_ids = tf.reshape(tf.gather(lang_ids, 1, axis=1), [-1, 1])
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
            attention_bias=attention_bias,
            training=training,
            encoder_padding=encoder_padding,
            enc_position=enc_position,
            vis=vis,
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
                            tf.shape(src), tf.fill(tf.shape(src), 0.5) # 0.5 can be finetune to change the probability of [C]
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
                    label_x=tgt_label,
                    src=src,
                    cross_lingual_id=cross_id,
                    model_fn=enc_fn,
                    lang_fn=self.lang_encoding,
                )
                clpm_plus = tf.stop_gradient(clpm_plus)

                clpm_check = tf.cast(
                    tf.greater(
                        tf.cast(self.optimizer.iterations.read_value(), tf.float32), 50000.0 # dont use [C] in the early 50000.0 steps
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
            decoder_self_attention_bias=decoder_self_attention_bias,
            attention_bias=attention_bias,
            training=training,
            cache=cache,
            dec_position=dec_position,
            vis=vis,
            decoder_padding=decoder_padding,
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
        if 5 in self.param["app"] and len(self.param["app"]) == 1:
            logits = run_encoder(src, src_id)
        if 2 in self.param["app"] and len(self.param["app"]) == 1:
            logits = run_decoder(tgt, tgt_id, None)
        if 3 in self.param["app"] and len(self.param["app"]) == 1:
            enc = run_encoder(src, src_id)
            logits = run_decoder(tgt, tgt_id, enc)
        if 1 in self.param["app"] and 2 in self.param["app"] and len(self.param["app"]) == 2:
            enc = run_encoder(
                src,
                src_id,
            )
            dec = run_decoder(tgt, tgt_id, None)
            logits = tf.concat([enc, dec], 0)
        return logits

    def call(self, inputs, training=False, **kwargs):
        vis = False
        if "vis" in kwargs:
            vis = kwargs["vis"]

        if training:
            src_id = kwargs["src_id"]
            tgt_id = kwargs["tgt_id"]
            if "tgt_label" in kwargs:
                tgt_label = kwargs["tgt_label"]
            else:
                tgt_label = None
            # span = kwargs["span"]
            # tgt_label = kwargs["tgt_label"]
            src, tgt = inputs[0], inputs[1]
            (
                attention_bias,
                decoder_self_attention_bias,
                encoder_padding,
                decoder_padding,
            ) = self.pre_processing(src, tgt)
            logits_raw = self.forward(
                src,
                tgt,
                training=training,
                attention_bias=attention_bias,
                decoder_self_attention_bias=decoder_self_attention_bias,
                encoder_padding=encoder_padding,
                decoder_padding=decoder_padding,
                vis=vis,
                src_id=src_id,
                tgt_id=tgt_id,
                tgt_label=tgt_label,
            )
            logits = self.probability_generator(logits_raw)
            return logits, logits_raw
        else:
            enc_lang = 1
            dec_lang = 1
            sos_id = self.param["SOS_ID"]
            eos_id = self.param["EOS_ID"]
            if "sos_id" in kwargs:
                sos_id = kwargs["sos_id"]
            if "eos_id" in kwargs:
                eos_id = kwargs["eos_id"]
            if "enc_lang" in kwargs:
                enc_lang = kwargs["enc_lang"]
            if "dec_lang" in kwargs:
                dec_lang = kwargs["dec_lang"]
            if "src_id" in kwargs:
                src_id = enc_lang = kwargs["src_id"]
            if "tgt_id" in kwargs:
                tgt_id = dec_lang = kwargs["tgt_id"]
            beam_size = 4
            if "beam_size" in kwargs:
                beam_size = kwargs["beam_size"]
            tgt = None
            src = inputs
            _, length = tf.unstack(tf.shape(src))
            enc_lang_token = tf.ones_like(src, dtype=tf.int32) * enc_lang
            cache, batch_size = self.prepare_cache(src, self.lang_encoding(enc_lang_token), sos_id)
            self.cache = cache
            if 5 in self.param["app"] and len(self.param["app"]) == 1:
                return cache.get("enc")
            if 1 in self.param["app"] and len(self.param["app"]) == 1:
                ids = self.probability_generator(cache.get("enc"))
                return tf.argmax(ids, -1)
            max_length = self.param["max_sequence_length"]
            dec_lang_token = (
                tf.expand_dims(tf.ones_like(cache["initial_ids"], dtype=tf.int32), 1) * dec_lang
            )
            autoregressive_fn = self.autoregressive_fn(
                max_length, lang_embedding=self.lang_encoding(dec_lang_token), beam_size=beam_size
            )
            if self.decoder.dynamic_dec != 0:
                re, score = self.predict(
                    autoregressive_fn,
                    eos_id=eos_id,
                    max_decode_length=int(length + 50),
                    cache=cache,
                    beam_size=beam_size,
                )
            else:
                re = self.probability_generator(cache["enc"])
            tf.print(
                "enc",
                self.encoder.dynamic_enc,
                "dec",
                self.decoder.dynamic_dec,
                output_stream=sys.stdout,
            )
            top_decoded_ids = re[:, 0, 1:]
            del cache, re, score
            return top_decoded_ids

    def seq2seq_update(self, x_logits, y_label, model_tape, **kwargs):
        # lang_1_domain = self.embedding_softmax_layer.domain_index[0]
        # lang_1_domain_seed = tf.keras.backend.random_bernoulli(
        #     tf.shape(lang_1_domain), tf.fill(tf.shape(lang_1_domain), 0.01)
        # )

        # lang_1 = tf.stop_gradient(
        #     self.embedding_softmax_layer._embedding(
        #         padding_util.seq_padding_remover(lang_1_domain, lang_1_domain_seed)
        #     ),
        #     0,
        # )

        # lang_2_domain = self.embedding_softmax_layer.domain_index[1]
        # lang_2_domain_seed = tf.keras.backend.random_bernoulli(
        #     tf.shape(lang_2_domain), tf.fill(tf.shape(lang_2_domain), 0.01)
        # )
        # lang_2 = tf.stop_gradient(
        #     self.embedding_softmax_layer._embedding(
        #         padding_util.seq_padding_remover(lang_2_domain, lang_2_domain_seed)
        #     ),
        #     0,
        # )
        # lang_1 = self.embedding_softmax_layer.discriminator(lang_1)
        # lang_2 = self.embedding_softmax_layer.discriminator(lang_2)
        # self.embedding_softmax_layer.de_domain(tf.reduce_mean(lang_1))
        # self.embedding_softmax_layer.en_domain(tf.reduce_mean(lang_2))
        # lang_1_label = tf.zeros_like(lang_1)
        # lang_2_label = tf.ones_like(lang_2)
        # # lang = tf.concat([lang_1,lang_2],0)
        # # lang_label = tf.concat([lang_1_label,lang_2_label],0)
        # binary_loss_lang_1 = tf.reduce_mean(
        #     tf.keras.losses.binary_crossentropy(lang_1_label, lang_1, label_smoothing=0.1)
        # )
        # binary_loss_lang_2 = tf.reduce_mean(
        #     tf.keras.losses.binary_crossentropy(lang_2_label, lang_2, label_smoothing=0.1)
        # )
        # binary_loss = binary_loss_lang_1 + binary_loss_lang_2

        # binary_loss =0
        loss = self.seq2seq_loss_FN([y_label, x_logits], auto_loss=False)
        model_gradients = model_tape.gradient(loss, self.trainable_variables)

        if self.param["clip_norm"] > 0:
            model_gradients, grad_norm = tf.clip_by_global_norm(
                model_gradients, self.param["clip_norm"]
            )
        else:
            grad_norm = tf.linalg.global_norm(model_gradients)
        self.optimizer.apply_gradients(zip(model_gradients, self.trainable_variables))
        self.grad_norm_ratio(grad_norm)
        self.perplexity(tf.math.exp(tf.cast(loss, tf.float32)))
        # if "tgt_label" in kwargs:
        #     y = kwargs["tgt_label"]
        if "tgt_metric" in kwargs:
            y_metric = kwargs["tgt_metric"]
        else:
            y_metric = y_label
        if "src_metric" in kwargs:
            src_metric = kwargs["src_metric"]
        else:
            src_metric = x_logits
        self.seq2seq_metric([y_metric, src_metric])
        # self.WEcos(self.embedding_softmax_layer)
        batch_size = tf.shape(x_logits)[0]
        self.tokenPerS(tf.cast(tf.math.multiply(batch_size, (tf.shape(x_logits)[1])), tf.float32))
        return

    def predict(
        self, autoregressive_fn, sos_id=1, eos_id=2, cache=None, beam_size=0, max_decode_length=99
    ):
        """Return predicted sequence."""
        decoded_ids, scores = self.beam_search.predict(
            autoregressive_fn,
            self.param["vocabulary_size"],
            eos_id=self.param["EOS_ID"],
            cache=cache,
            max_decode_length=max_decode_length,
            beam_size=beam_size,
        )
        return decoded_ids, scores

    def get_config(self):
        c = self.param
        return c
