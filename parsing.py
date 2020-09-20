import random
import re
from itertools import chain
from collections import defaultdict
from tqdm import trange

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from IPython.display import clear_output
from matplotlib import pyplot as plt

# data

ROOT = "[ROOT]"
ROOT_ARC = "root"
ROOT_POS = "[ROOT]"


class Example:
    def __init__(self, tokens, arcs, pos):
        self.tokens = tokens
        self.arcs = arcs
        self.pos = pos

    @property
    def num_tokens(self):
        return len(self.tokens)


def load_examples(path, arc2id=None, pos2id=None):
    examples = []
    num_strange = 0
    arc_names = set()
    pos_names = {ROOT_POS}

    with open(path) as f:
        tokens_i = []
        arcs_i = []
        pos_i = []
        flag_strange = False
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                if line.startswith("#"):
                    continue
                features = line.split("\t")
                token_id = features[0]
                token = features[1]
                pos = features[3]
                dependent = features[6]
                arc_name = features[7]
                arc_names.add(arc_name)
                pos_names.add(pos)
                try:
                    arc = int(token_id), int(dependent), arc_name
                    tokens_i.append(token)
                    arcs_i.append(arc)
                    pos_i.append(pos)
                except:
                    flag_strange = True
                    num_strange += 1
            else:
                if tokens_i and arcs_i:
                    if flag_strange:
                        flag_strange = False
                    else:
                        tokens_i.insert(0, ROOT)
                        arcs_i.insert(0, (0, 0, ROOT_ARC))
                        pos_i.insert(0, ROOT_POS)
                        example = Example(tokens_i.copy(), arcs_i.copy(), pos_i.copy())
                        examples.append(example)
                    tokens_i.clear()
                    arcs_i.clear()
                    pos_i.clear()

    random.seed(228)
    random.shuffle(examples)
    print(f"num examples: {len(examples)}")
    print(f"num strange: {num_strange}")
    print(f"num arc types: {len(arc_names)}")
    if arc2id is None:
        arc2id = {x: i for i, x in enumerate(arc_names)}

    if pos2id is None:
        pos2id = {x: i for i, x in enumerate(pos_names)}

    for example in examples:
        example.arcs = [(x[0], x[1], arc2id[x[2]]) for x in example.arcs]
        example.pos = [pos2id[x] for x in example.pos]

    for ex in examples:
        assert ex.num_tokens > 0
        assert len(ex.arcs) == ex.num_tokens
        assert len(ex.pos) == ex.num_tokens
        n_arcs = len(ex.arcs)
        x0 = []
        for arc in ex.arcs:
            assert 0 <= arc[1] < n_arcs
            assert arc[2] in arc2id.values()
            x0.append(arc[0])
        assert x0 == list(range(n_arcs))
        n_pos = len(pos2id)
        for pos in ex.pos:
            assert 0 <= pos < n_pos
            assert pos in pos2id.values()

    return examples, arc2id, pos2id


# model

class DependencyParser:
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config

        self.tokens_ph = None
        self.sequence_len_ph = None
        self.pos_ph = None
        self.type_ids_ph = None
        self.training_ph = None

        self.s_arc = None
        self.s_type = None

        self.pos_log_probs = None
        self.pos_preds = None

        self.loss = None
        self.train_op = None
        self.acc_op = None
        self.reset_op = None

        # for debug
        self.global_step = None
        self.all_are_finite = None

    def build(self):
        self._set_placeholders()

        emb_config = self.config["emb"]
        elmo = hub.Module(emb_config["elmo_dir"], trainable=False)
        input_dict = {
            "tokens": self.tokens_ph,
            "sequence_len": self.sequence_len_ph
        }
        x = elmo(input_dict, signature="tokens", as_dict=True)["elmo"]

        elmo_dropout = tf.keras.layers.Dropout(emb_config["elmo_dropout"])
        x = elmo_dropout(x, training=self.training_ph)

        sequence_mask = tf.sequence_mask(self.sequence_len_ph)
        sequence_mask = tf.cast(sequence_mask, tf.float32)

        with tf.variable_scope("dependency_parser"):
            if emb_config["use_pos"]:
                pos_emb = tf.keras.layers.Embedding(
                    input_dim=emb_config["num_pos_labels"],
                    output_dim=emb_config["pos_emb_dim"]
                )
                x_pos = pos_emb(self.pos_ph)

                pos_dropout = tf.keras.layers.Dropout(emb_config["pos_emb_dropout"])
                x_pos = pos_dropout(x_pos, training=self.training_ph)

                if emb_config["concat"]:
                    x = tf.concat([x, x_pos], axis=-1)
                else:
                    x += x_pos

            emb_dropout = tf.keras.layers.Dropout(emb_config["combined_dropout"])
            x = emb_dropout(x, training=self.training_ph)

            if emb_config["layernorm"]:
                emb_layernorm = tf.keras.layers.LayerNormalization()
                x = emb_layernorm(x)

            attn_config = self.config["attention"]
            d_model = attn_config["num_heads"] * attn_config["head_dim"]
            x = tf.keras.layers.Dense(d_model)(x)
            for i in range(attn_config["num_layers"]):
                attn = DotProductAttention(**attn_config)
                x = attn(x, training=self.training_ph, mask=sequence_mask)

            parser_config = self.config["parser"]
            parser = ParserHead(parser_config)
            self.s_arc, self.s_type = parser(x, training=self.training_ph, mask=sequence_mask)

        self._set_loss()
        self._set_train_op()
        self.sess.run(tf.global_variables_initializer())

    def train(self, train_examples, eval_examples, num_epochs=1, batch_size=128, plot_step=10, plot_train_steps=1000):
        train_loss = []
        eval_loss = []
        eval_las = []
        eval_uas = []

        def plot():
            clear_output()
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
            ax1.set_title("train loss")
            ax1.plot(train_loss[-plot_train_steps:])
            ax1.grid()

            ax2.set_title("eval loss")
            ax2.plot(eval_loss, marker='o')
            ax2.grid()

            ax3.set_title("greedy attachment scores")
            ax3.plot(eval_las, marker='o', label='las')
            ax3.plot(eval_uas, marker='o', label='uas')
            ax3.legend()
            ax3.grid()

            plt.show()

        num_acc_steps = self.config["opt"]["num_accumulation_steps"]
        global_batch_size = batch_size * num_acc_steps
        epoch_steps = len(train_examples) // global_batch_size + 1
        num_train_steps = num_epochs * epoch_steps

        print(f"global batch size: {global_batch_size}")
        print(f"epoch steps: {epoch_steps}")
        print(f"num_train_steps: {num_train_steps}")

        for step in range(num_train_steps):
            if self.config["opt"]["num_accumulation_steps"] == 1:
                examples_batch = random.sample(train_examples, batch_size)
                feed_dict = self._get_feed_dict(examples_batch, training=True)
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                train_loss.append(loss)
                print(loss)
            else:
                # обнуление переменных, хранящих накопленные градиенты
                self.sess.run(self.reset_op)
                # with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                #     v = tf.get_variable("dependency_parser/dense/kernel/accum").eval(session=self.sess)
                #     print("init:")
                #     print(v)
                losses_tmp = []
                aaf = True

                # накопление градиентов
                for _ in range(num_acc_steps):
                    examples_batch = random.sample(train_examples, batch_size)
                    feed_dict = self._get_feed_dict(examples_batch, training=True)
                    _, loss, gs, aaf_step = self.sess.run([self.acc_op, self.loss, self.global_step, self.all_are_finite], feed_dict=feed_dict)
                    print(gs, loss, aaf_step)
                    # with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                    #     v = tf.get_variable("dependency_parser/dense/kernel/accum").eval(session=self.sess)
                    #     print("accum:")
                    #     print(v)
                    losses_tmp.append(loss)
                    aaf &= aaf_step

                # проверка финитности градиентов
                if not aaf:
                    while True:
                        examples_batch = random.sample(train_examples, batch_size)
                        feed_dict = self._get_feed_dict(examples_batch, training=True)
                        _, loss, gs, aaf_step = self.sess.run([self.acc_op, self.loss, self.global_step, self.all_are_finite], feed_dict=feed_dict)
                        if aaf_step:
                            break

                # обновление весов
                self.sess.run(self.train_op)
                train_loss.append(np.mean(losses_tmp))

            if step % plot_step == 0:
                plot()

            if step != 0 and step % epoch_steps == 0:
                losses_tmp = []
                las_tmp = []
                uas_tmp = []
                for start in range(0, len(eval_examples), batch_size):
                    end = start + batch_size
                    examples_batch = eval_examples[start:end]
                    feed_dict = self._get_feed_dict(examples_batch, training=False)
                    loss, s_arc, s_type = self.sess.run([self.loss, self.s_arc, self.s_type], feed_dict=feed_dict)
                    losses_tmp.append(loss)

                    s_arc_argmax = s_arc.argmax(-1)  # [N, T]
                    s_type_argmax = s_type.argmax(-1)  # [N, T, T]
                    for i, x in enumerate(examples_batch):
                        for arc in x.arcs:
                            j = arc[0]
                            head_true_ij = arc[1]
                            type_true_ij = arc[2]
                            head_pred_ij = s_arc_argmax[i, j]
                            flag_arc = head_true_ij == head_pred_ij
                            flag_type = s_type_argmax[i, j, head_pred_ij] == type_true_ij
                            las_tmp.append(flag_arc & flag_type)
                            uas_tmp.append(flag_arc)

                eval_loss.append(np.mean(losses_tmp))
                eval_las.append(np.mean(las_tmp))
                eval_uas.append(np.mean(uas_tmp))
                plot()
        plot()

    def predict(self, examples, batch_size=128):
        y_pred = []
        for start in trange(0, len(examples), batch_size):
            end = start + batch_size
            examples_batch = examples[start:end]
            y_pred += self._predict_batch(examples_batch)
        return y_pred

    @staticmethod
    def evaluate(y_true, y_pred):
        flags_las = []
        flags_uas = []
        for i, j in zip(y_true, y_pred):
            flags_las.append(i == j)
            flags_uas.append(i[:-1] == j[:-1])
        las = np.mean(flags_las)
        uas = np.mean(flags_uas)
        return las, uas

    def _predict_batch(self, examples):
        feed_dict = self._get_feed_dict(examples, training=False)
        s_arc, s_type = self.sess.run([self.s_arc, self.s_type], feed_dict=feed_dict)
        preds = []
        for i in range(len(examples)):
            length_i = examples[i].num_tokens
            s_arc_i = s_arc[i, :length_i, :length_i]
            s_type_i = s_type[i, :length_i, :length_i]

            indices_dep = range(length_i)
            indices_head = mst(s_arc_i)
            indices_type = s_type_i[indices_dep, indices_head].argmax(-1)

            preds_i = list(zip(indices_head, indices_type))
            preds.append(preds_i)
        return preds

    def _get_feed_dict(self, examples, training):
        PAD = "[PAD]"
        tokens = [x.tokens for x in examples]
        sequence_len = [x.num_tokens for x in examples]
        maxlen = max(sequence_len)
        tokens = [x + [PAD] * (maxlen - l) for x, l in zip(tokens, sequence_len)]
        type_ids = [(i, *y) for i, x in enumerate(examples) for y in x.arcs]
        feed_dict = {
            self.tokens_ph: tokens,
            self.sequence_len_ph: sequence_len,
            self.type_ids_ph: type_ids,
            self.training_ph: training
        }
        if self.config["emb"]["use_pos"]:
            pos_id_pad = self.config["emb"]["num_pos_labels"]
            pos = [x.pos + [pos_id_pad] * (maxlen - l) for x, l in zip(examples, sequence_len)]
            feed_dict[self.pos_ph] = pos
        return feed_dict

    def _set_placeholders(self):
        self.tokens_ph = tf.placeholder(tf.string, shape=[None, None], name="tokens_ph")
        self.sequence_len_ph = tf.placeholder(tf.int32, shape=[None], name="sequence_len_ph")
        self.pos_ph = tf.placeholder(tf.int32, shape=[None, None], name="pos_ph")
        self.type_ids_ph = tf.placeholder(tf.int32, shape=[None, 4],
                                          name="type_ids_ph")  # [id_example, id_dep, id_head, id_label]
        self.training_ph = tf.placeholder(tf.bool, shape=None, name="training_ph")

    def _set_loss(self):
        arc_ids = self.type_ids_ph[:, :-1]
        s_arc_gather = tf.gather_nd(self.s_arc, arc_ids)
        s_type_gather = tf.gather_nd(self.s_type, self.type_ids_ph)
        arc_loss = -tf.reduce_mean(tf.log(s_arc_gather))
        type_loss = -tf.reduce_mean(tf.log(s_type_gather))
        self.loss = arc_loss + type_loss

    def _set_train_op(self):
        if self.config["opt"]["num_accumulation_steps"] == 1:
            self._set_train_op_wo_acc()
        else:
            self._set_train_op_with_acc()

    def _set_train_op_wo_acc(self):
        tvars = tf.trainable_variables()
        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(
            init_lr=self.config["opt"]["init_lr"],
            global_step=global_step,
            warmup_steps=self.config["opt"]["warmup_steps"]
        )
        optimizer = tf.train.AdamOptimizer(lr)
        grads = tf.gradients(self.loss, tvars)
        if self.config["opt"]["clip_grads"]:
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.config["opt"]["clip_norm"])
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    def _set_train_op_with_acc(self):
        tvars = tf.trainable_variables()
        accum_vars = [
            tf.get_variable(
                name=v.name.split(":")[0] + "/accum",
                shape=v.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer()
            ) for v in tvars
        ]
        self.global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(init_lr=self.config["opt"]["init_lr"], global_step=self.global_step, warmup_steps=self.config["opt"]["warmup_steps"])
        optimizer = tf.train.AdamOptimizer(lr)
        num_acc_steps = self.config["opt"]["num_accumulation_steps"] * 1.0
        grads = tf.gradients(self.loss / num_acc_steps, tvars)
        self.all_are_finite = tf.reduce_all([tf.reduce_all(tf.is_finite(g)) for g in grads])
        if self.config["opt"]["clip_grads"]:
            grads, _ = tf.clip_by_global_norm(
                grads,
                clip_norm=self.config["opt"]["clip_norm"],
                use_norm=tf.cond(
                    self.all_are_finite,
                    lambda: tf.global_norm(grads),
                    lambda: tf.constant(1.0)
                )
            )
        self.reset_op = [v.assign(tf.zeros_like(v)) for v in accum_vars]
        self.acc_op = [v.assign_add(g) for v, g in zip(accum_vars, grads)]
        self.train_op = optimizer.apply_gradients(zip(accum_vars, tvars), global_step=self.global_step)
        with tf.control_dependencies([self.train_op]):
            self.global_step.assign_add(1)


class ParserHead(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()

        config_mlp = config["mlp"]
        config_arc = config["arc"]
        config_type = config["type"]

        self.mlp_arc_d = MLP(config_mlp["num_layers"], config_arc["hidden_dim"], tf.nn.relu, config_mlp["dropout"])
        self.mlp_arc_h = MLP(config_mlp["num_layers"], config_arc["hidden_dim"], tf.nn.relu, config_mlp["dropout"])
        self.mlp_type_d = MLP(config_mlp["num_layers"], config_type["hidden_dim"], tf.nn.relu, config_mlp["dropout"])
        self.mlp_type_h = MLP(config_mlp["num_layers"], config_type["hidden_dim"], tf.nn.relu, config_mlp["dropout"])

        self.biaffine_attention = BiAffineAttention(config_arc["hidden_dim"], config_arc["hidden_dim"])
        self.bilinear = BiLinear(config_type["hidden_dim"], config_type["hidden_dim"], config_type["num_labels"])

    def call(self, x, training=False, mask=None):
        # head / dependent projections
        type_d = self.mlp_type_d(x, training)  # [N, T, type_dim], dependent
        type_h = self.mlp_type_h(x, training)  # [N, T, type_dim], head

        arc_d = self.mlp_arc_d(x, training)  # [N, T, arc_dim], dependent
        arc_h = self.mlp_arc_h(x, training)  # [N, T, arc_dim], head

        # type scores
        s_type = self.bilinear(type_d, type_h)  # [N, T, T, num_arc_labels]
        s_type = tf.nn.softmax(s_type, axis=-1)

        # arc scores
        s_arc = self.biaffine_attention(arc_d, arc_h)  # [N, T, T]
        mask = tf.expand_dims(mask, [-1])  # [N, T, 1]
        s_arc += (1. - mask) * -1e9
        s_arc = tf.nn.softmax(s_arc, axis=-1)
        return s_arc, s_type


def add_ones(x):
    ones = tf.ones_like(x[..., :1])
    x = tf.concat([x, ones], axis=-1)
    return x


class MLP(tf.keras.layers.Layer):
    def __init__(self, num_layers, hidden_dim, activation, dropout):
        super().__init__()
        self.dense_layers = [tf.keras.layers.Dense(hidden_dim, activation=activation) for _ in range(num_layers)]
        self.dropout_layers = [tf.keras.layers.Dropout(dropout) for _ in range(num_layers)]

    def call(self, x, training=False):
        for dense, dropout in zip(self.dense_layers, self.dropout_layers):
            x = dense(x)
            x = dropout(x, training=training)
        return x


class BiLinear(tf.keras.layers.Layer):
    def __init__(self, left_dim, right_dim, output_dim):
        super().__init__()
        self.w = tf.get_variable("w", shape=(output_dim, left_dim + 1, right_dim + 1), dtype=tf.float32)
        self.u = tf.get_variable("u", shape=(left_dim, output_dim), dtype=tf.float32)
        self.v = tf.get_variable("v", shape=(right_dim, output_dim), dtype=tf.float32)

    def call(self, x_left, x_right):
        x_left_1 = add_ones(x_left)  # [N, T, left_dim + 1]
        x_right_1 = add_ones(x_right)  # [N, T, right_dim + 1]
        x_right_1_t = tf.transpose(x_right_1, [0, 2, 1])  # [N, right_dim + 1, T]
        x_left_u = tf.matmul(x_left, self.u)  # [N, T, output_dim]
        x_right_v = tf.matmul(x_right, self.v)  # [N, T, output_dim]
        scores = tf.expand_dims(x_left_1, [1]) @ self.w @ tf.expand_dims(x_right_1_t, [1])  # [N, output_dim, T, T]
        scores = tf.transpose(scores, [0, 2, 3, 1])  # [N, T, T, output_dim]
        scores += tf.expand_dims(x_left_u, [2])
        scores += tf.expand_dims(x_right_v, [2])
        return scores


class BiAffineAttention(tf.keras.layers.Layer):
    def __init__(self, left_dim, right_dim):
        super().__init__()
        self.w = tf.get_variable("W", shape=(left_dim + 1, right_dim + 1), dtype=tf.float32)

    def call(self, x_left, x_right):
        """
        x_left - tf.Tensor of shape [N, T, left_dim]
        x_right - tf.Tensor of shape [N, T, left_right]
        return: x - tf.Tensor of shape [N, T, T]
        """
        x_left_1 = add_ones(x_left)  # [N, T, left_dim + 1]
        x_right_1 = add_ones(x_right)  # [N, T, right_dim + 1]
        x_right_1_t = tf.transpose(x_right_1, [0, 2, 1])  # [N, right_dim + 1, T]
        x = x_left_1 @ self.w @ x_right_1_t  # [N, T, T]
        return x


class DotProductAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        d_model = kwargs["num_heads"] * kwargs["head_dim"]
        self.mha = MHA(**kwargs)
        self.dense_ff = tf.keras.layers.Dense(kwargs["dff"], activation=tf.nn.relu)
        self.dense_model = tf.keras.layers.Dense(d_model)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.dropout_rc1 = tf.keras.layers.Dropout(kwargs["dropout_rc"])
        self.dropout_rc2 = tf.keras.layers.Dropout(kwargs["dropout_rc"])
        self.dropout_ff = tf.keras.layers.Dropout(kwargs["dropout_ff"])

    def call(self, x, training=False, mask=None):
        x1 = self.mha(x, mask=mask)
        x1 = self.dropout_rc1(x1, training=training)
        x = self.ln1(x + x1)
        x1 = self.dense_ff(x)
        x1 = self.dropout_ff(x1, training=training)
        x1 = self.dense_model(x1)
        x1 = self.dropout_rc2(x1, training=training)
        x = self.ln2(x + x1)
        return x


class MHA(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_heads = kwargs["num_heads"]
        self.head_dim = kwargs["head_dim"]
        self.dense_input = tf.keras.layers.Dense(self.num_heads * self.head_dim * 3)

    def call(self, x, mask=None):
        """
        https://arxiv.org/abs/1706.03762
        :param x: tf.Tensor of shape [N, T, H * D]
        :param mask: tf.Tensor of shape [N, T]
        :return: tf.Tensor of shape [N, T, H * D]
        """
        batch_size = tf.shape(x)[0]
        qkv = self.dense_input(x)  # [N, T, H * D * 3]
        qkv = tf.reshape(qkv, [batch_size, -1, self.num_heads, self.head_dim, 3])  # [N, T, H, D, 3]
        qkv = tf.transpose(qkv, [4, 0, 2, 1, 3])  # [3, N, H, T, D]
        q, k, v = tf.unstack(qkv)  # 3 * [N, H, T, D]

        logits = tf.matmul(q, k, transpose_b=True)  # [N, H, T, T]
        logits /= self.head_dim ** 0.5  # [N, H, T, T]

        mask = mask[:, None, :, None]
        logits += (1. - mask) * -1e9

        w = tf.nn.softmax(logits, axis=-1)  # [N, H, T, T] (k-axis)
        x = tf.matmul(w, v)  # [N, H, T, D]
        x = tf.transpose(x, [0, 2, 1, 3])  # [N, T, H, D]
        x = tf.reshape(x, [batch_size, -1, self.num_heads * self.head_dim])  # [N, T, D * H]
        return x


def mst(scores, eps=1e-10):
    """
    Chu-Liu-Edmonds' algorithm for finding minimum spanning arborescence in graphs.
    Calculates the arborescence with node 0 as root.
    :param scores: `scores[i][j]` is the weight of edge from node `j` to node `i`.
    :returns an array containing the head node (node with edge pointing to current node) for each node,
             with head[0] fixed as 0

    1. удалить рёбра вида (*, r)
    2. из пары рёбер ((i, j), (j, i)) выбать ребро с минимальным весом
    3. для каждой вершины child находим вершину parent с минимальным весом
    4. если граф ацикличный, то конец
    5. иначе,
    """
    length = scores.shape[0]
    scores = scores * (1 - np.eye(length))  # mask all the diagonal elements wih a zero
    heads = np.argmax(scores, axis=1)  # THIS MEANS THAT scores[i][j] = score(j -> i)!
    heads[0] = 0  # the root has a self-loop to make it special
    tokens = np.arange(1, length)
    roots = np.where(heads[tokens] == 0)[0] + 1
    if len(roots) < 1:
        root_scores = scores[tokens, 0]
        head_scores = scores[tokens, heads[tokens]]
        new_root = tokens[np.argmax(root_scores / (head_scores + eps))]
        heads[new_root] = 0
    elif len(roots) > 1:
        root_scores = scores[roots, 0]
        scores[roots, 0] = 0
        new_heads = np.argmax(scores[roots][:, tokens], axis=1) + 1
        new_root = roots[np.argmin(scores[roots, new_heads] / (root_scores + eps))]
        heads[roots] = new_heads
        heads[new_root] = 0

    edges = defaultdict(set)  # head -> dep
    vertices = set((0,))
    for dep, head in enumerate(heads[tokens]):
        vertices.add(dep + 1)
        edges[head].add(dep + 1)
    for cycle in _find_cycle(vertices, edges):
        dependents = set()
        to_visit = set(cycle)
        while len(to_visit) > 0:
            node = to_visit.pop()
            if node not in dependents:
                dependents.add(node)
                to_visit.update(edges[node])
        cycle = np.array(list(cycle))
        old_heads = heads[cycle]
        old_scores = scores[cycle, old_heads]
        non_heads = np.array(list(dependents))
        scores[np.repeat(cycle, len(non_heads)), np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
        new_heads = np.argmax(scores[cycle][:, tokens], axis=1) + 1
        new_scores = scores[cycle, new_heads] / (old_scores + eps)
        change = np.argmax(new_scores)
        changed_cycle = cycle[change]
        old_head = old_heads[change]
        new_head = new_heads[change]
        heads[changed_cycle] = new_head
        edges[new_head].add(changed_cycle)
        edges[old_head].remove(changed_cycle)
    return heads


def _find_cycle(vertices, edges):
    """
    https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm  # NOQA
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/etc/tarjan.py  # NOQA
    """
    _index = [0]
    _stack = []
    _indices = {}
    _lowlinks = {}
    _onstack = defaultdict(lambda: False)
    _SCCs = []

    def _strongconnect(v):
        _indices[v] = _index[0]
        _lowlinks[v] = _index[0]
        _index[0] += 1
        _stack.append(v)
        _onstack[v] = True

        for w in edges[v]:
            if w not in _indices:
                _strongconnect(w)
                _lowlinks[v] = min(_lowlinks[v], _lowlinks[w])
            elif _onstack[w]:
                _lowlinks[v] = min(_lowlinks[v], _indices[w])

        if _lowlinks[v] == _indices[v]:
            SCC = set()
            while True:
                w = _stack.pop()
                _onstack[w] = False
                SCC.add(w)
                if not (w != v):
                    break
            _SCCs.append(SCC)

    for v in vertices:
        if v not in _indices:
            _strongconnect(v)

    return [SCC for SCC in _SCCs if len(SCC) > 1]


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def _apply_gradients(self, grads_and_vars, learning_rate):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                    tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                    tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                              tf.square(grad)))
            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self.weight_decay_rate > 0:
                if self._do_use_weight_decay(param_name):
                    update += self.weight_decay_rate * param

            update_with_lr = learning_rate * update
            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])

        return assignments

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        if isinstance(self.learning_rate, dict):
            key_to_grads_and_vars = {}
            for grad, var in grads_and_vars:
                update_for_var = False
                for key in self.learning_rate:
                    if key in var.name:
                        update_for_var = True
                        if key not in key_to_grads_and_vars:
                            key_to_grads_and_vars[key] = []
                        key_to_grads_and_vars[key].append((grad, var))
                if not update_for_var:
                    raise ValueError("No learning rate specified for variable", var)
            assignments = []
            for key, key_grads_and_vars in key_to_grads_and_vars.items():
                assignments += self._apply_gradients(key_grads_and_vars,
                                                     self.learning_rate[key])
        else:
            assignments = self._apply_gradients(grads_and_vars, self.learning_rate)
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
