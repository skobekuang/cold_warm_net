import random
import numpy as np
from collections import Counter
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from deepctr.feature_column import build_input_features, create_embedding_matrix
from deepctr.layers import PredictionLayer, DNN, combined_dnn_input
from deepmatch.inputs import input_from_feature_columns
from deepmatch.layers.core import Similarity


def gen_data_set(data, negsample=0):
    data.sort_values("timestamp", inplace=True)
    item_ids = data['movie_id'].unique()

    train_set = []
    test_set = []
    for reviewer_id, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['movie_id'].tolist()
        rating_list = hist['rating'].tolist()

        if negsample > 0:
            neg_list = np.random.choice(item_ids, size=len(pos_list) * negsample, replace=True)
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            if i != len(pos_list) - 1:
                train_set.append((reviewer_id, hist[::-1], pos_list[i], 1, len(hist[::-1]), rating_list[i]))
                for negi in range(negsample):
                    train_set.append((reviewer_id, hist[::-1], neg_list[i * negsample + negi], 0, len(hist[::-1])))
            else:
                test_set.append((reviewer_id, hist[::-1], pos_list[i], 1, len(hist[::-1]), rating_list[i]))

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(len(train_set[0]), len(test_set[0]))

    return train_set, test_set


def gen_pred_set(data):
    data.sort_values("timestamp", inplace=True)
    item_ids = data['movie_id'].unique()

    train_set = []
    for reviewer_id, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['movie_id'].tolist()
        rating_list = hist['rating'].tolist()
        hist = pos_list[:-1]
        train_set.append((reviewer_id, hist[::-1], pos_list, 1, len(hist[::-1]), rating_list))

    random.shuffle(train_set)

    print(len(train_set[0]))

    return train_set


def gen_model_input(train_set, user_profile, seq_max_len):
    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])

    train_label_v1 = train_label
    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)

    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad,
                         "hist_len": train_hist_len, 'train_label_v1': train_label_v1}

    for key in ["gender", "age", "occupation", "zip", 'user_counts', 'new_old', 'user_group']:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input, train_label


def meta_ban_bias(user_feature_columns, item_feature_columns, user_group_embs, user_dnn_hidden_units=(64, 32),
                   item_dnn_hidden_units=(64, 32), gate_dnn_hidden_units=(64, 32), bias_dnn_hidden_units=(64, 32),
                   embedding_dim=32, dnn_activation='tanh', gate_dnn_activation='relu', dnn_use_bn=False,
                   output_activation='linear', l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001,
                   seed=1024, metric='cos', gamma=10):
    """Instantiates the Deep Structured Semantic Model architecture.

    :param user_feature_columns: An iterable containing user's features used by  the model.
    :param item_feature_columns: An iterable containing item's features used by  the model.
    :param user_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of user tower
    :param item_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of item tower
    :param dnn_activation: Activation function to use in deep net
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param seed: integer ,to use as random seed.
    :param metric: str, ``"cos"`` for  cosine  or  ``"ip"`` for inner product
    :return: A Keras model instance.

    """
    tf.reset_default_graph()

    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding,
                                                    seed=seed, seq_mask_zero=True)

    train_label_v1 = tf.keras.Input(shape=(1,), name='train_label_v1')

    hash_u_bucket = user_group_embs.shape[0]
    bucket_emb = tf.get_variable(name='bucket_emb', initial_value=user_group_embs, trainable=False,
                             shape=[hash_u_bucket, embedding_dim], dtype=tf.float32)

    # Cold expert
    user_features = build_input_features(user_feature_columns)
    user_inputs_list = list(user_features.values())
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features,
                                                                                   user_feature_columns,
                                                                                   l2_reg_embedding, seed=seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    length = len(user_sparse_embedding_list)

    user_cold_dnn_input = combined_dnn_input(user_sparse_embedding_list[:length - 1], user_dense_value_list)

    user_cold_dnn_out = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                            dnn_use_bn, output_activation='linear', seed=seed)(user_cold_dnn_input)

    # Cold attention
    user_att_dnn_hidden_units = [hash_u_bucket]
    att_dnn_activation = 'relu'
    user_cold_hash_att = DNN(user_att_dnn_hidden_units, att_dnn_activation, l2_reg_dnn, dnn_dropout,
                             dnn_use_bn, output_activation='linear', seed=seed)(user_cold_dnn_input)
    user_cold_hash_att = tf.nn.softmax(user_cold_hash_att, axis=-1, name='cold_user_attention_softmax')

    cold_bucket_dnn = tf.matmul(user_cold_hash_att, bucket_emb)

    user_cold_dnn_out = tf.concat([user_cold_dnn_out, cold_bucket_dnn], axis=-1)

    merge_dnn_hidden_units = [embedding_dim]

    user_cold_dnn_out = DNN(merge_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                            dnn_use_bn, seed=seed)(user_cold_dnn_out)

    print("user_cold_dnn_out_att shape: {}".format(user_cold_dnn_out.shape))

    # Warm expert
    user_warm_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

    user_warm_dnn_out = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                            dnn_use_bn, output_activation='linear', seed=seed)(user_warm_dnn_input)

    # Warm attention
    user_warm_hash_att = DNN(user_att_dnn_hidden_units, att_dnn_activation, l2_reg_dnn, dnn_dropout,
                             dnn_use_bn, output_activation='linear', seed=seed)(user_warm_dnn_input)

    user_warm_hash_att = tf.nn.softmax(user_warm_hash_att, axis=-1, name='warm_user_attention_softmax')

    warm_bucket_dnn = tf.matmul(user_warm_hash_att, bucket_emb)

    user_warm_dnn_out = tf.concat([user_warm_dnn_out, warm_bucket_dnn], axis=-1)

    user_warm_dnn_out = DNN(merge_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                            dnn_use_bn, seed=seed)(user_warm_dnn_out)

    print("user_warm_dnn_out_att shape: {}".format(user_warm_dnn_out.shape))

    # Gate network
    gate_dnn_input = combined_dnn_input(user_sparse_embedding_list[1:length - 1], user_dense_value_list)

    gate_dnn_out = DNN(gate_dnn_hidden_units, gate_dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, seed=seed)(gate_dnn_input)

    EXPERT_NUM = 2

    gate_attention_units = (embedding_dim, EXPERT_NUM)
    gate_attention_activation = 'linear'
    gate_attention = DNN(gate_attention_units, gate_attention_activation, l2_reg_dnn, dnn_dropout,
                         dnn_use_bn, seed=seed)(gate_dnn_out)

    concat_cold_warm = tf.concat([user_cold_dnn_out, user_warm_dnn_out], axis=1)
    print("[test matmul1] {} {} {}".format(
        user_cold_dnn_out.shape, user_warm_dnn_out.shape, concat_cold_warm.shape))

    concat_cold_warm_reshape = tf.reshape(concat_cold_warm, [-1, EXPERT_NUM, embedding_dim],
                                          name='concat_cold_warm_reshape')
    gate_attention_reshape = tf.reshape(tf.nn.softmax(gate_attention, axis=-1), [-1, 1, EXPERT_NUM],
                                        name='gate_attention_softmax')

    user_output_before = tf.matmul(gate_attention_reshape, concat_cold_warm_reshape)
    user_dnn_out = tf.reshape(user_output_before, [-1, embedding_dim], name='user_dnn_out')
    print("[test matmul] {} {} {} {}".format(gate_attention_reshape.shape, concat_cold_warm_reshape.shape,
                                             user_output_before.shape, user_dnn_out.shape))

    # bias tower
    bias_dnn_out = DNN(bias_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, seed=seed)(gate_dnn_input)

    logits_bias = tf.keras.layers.Dense(1, activation='linear', name='logits_bias_layer')(bias_dnn_out)

    bias_score = tf.identity(logits_bias, name='pred_bias')
    print('bias_score shape:{}'.format(bias_score))

    # Item tower
    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())
    item_sparse_embedding_list, item_dense_value_list = (
            input_from_feature_columns(item_features, item_feature_columns,
                                       l2_reg_embedding, seed=seed,
                                       embedding_matrix_dict=embedding_matrix_dict))
    item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)

    item_dnn_out = DNN(item_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, output_activation='linear', seed=seed)(item_dnn_input)

    print('user_output shape: {} {} {}'.format(user_dnn_out.shape,
                                               item_dnn_out.shape, concat_cold_warm_reshape.shape))

    score_cold = Similarity(type=metric, gamma=gamma)([user_cold_dnn_out, item_dnn_out])
    score_warm = Similarity(type=metric, gamma=gamma)([user_warm_dnn_out, item_dnn_out])
    score = Similarity(type=metric, gamma=gamma)([user_dnn_out, item_dnn_out])

    pred = PredictionLayer("binary", False)(score)
    pred_cold = PredictionLayer("binary", False)(score_cold)
    pred_warm = PredictionLayer("binary", False)(score_warm)
    pred_warm_no_gradient = tf.stop_gradient(pred_warm, name='pred_what_no_gradient')
    print('pred shape: {} {} {}'.format(pred.shape, pred_cold.shape, pred_warm.shape))

    gate_softmax_warm_weight = tf.square(gate_attention_reshape[:, 0, 1])
    use_warm_loss_before = tf.reshape(gate_softmax_warm_weight, [-1, 1])

    use_warm_loss = tf.keras.layers.Dense(1, activation='linear', name='use_warm_loss_layer')(use_warm_loss_before)

    print("use_warm_loss: {}".format(use_warm_loss.shape))

    soft_loss = K.mean(K.binary_crossentropy(pred_warm_no_gradient, pred_cold), axis=-1)
    cold_loss = K.mean(K.binary_crossentropy(train_label_v1, pred_cold), axis=-1)
    warm_loss = K.mean(K.binary_crossentropy(train_label_v1, pred_warm), axis=-1)

    # logits_bias loss
    logits_bias_loss = K.mean(K.square(logits_bias), axis=-1)

    COLD_LOSS_BOOST = 3.0
    soft_loss_v2 = tf.where(tf.less(cold_loss, warm_loss), COLD_LOSS_BOOST * cold_loss, soft_loss)

    model = Model(inputs=[user_inputs_list + item_inputs_list + [train_label_v1]],
                  outputs=[pred, soft_loss_v2, use_warm_loss, logits_bias_loss])

    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("user_embedding", user_dnn_out)
    model.__setattr__("item_embedding", item_dnn_out)

    return model


if __name__ == '__main__':
    data_path = "./"

    unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    user = pd.read_csv(data_path + 'ml-1m/users.dat', sep='::', header=None, names=unames)
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(data_path + 'ml-1m/ratings.dat', sep='::', header=None, names=rnames)
    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(data_path + 'ml-1m/movies.dat', sep='::', header=None, names=mnames)

    data = pd.merge(pd.merge(ratings, movies), user)

    user_id_list = data['user_id']
    user_dict = Counter(user_id_list)
    user_counts = pd.DataFrame.from_dict(user_dict, orient='index', columns=['user_counts'])
    user_counts = user_counts.reset_index().rename(columns={'index': 'user_id'})
    data = pd.merge(data, user_counts)
    data['new_old'] = 1
    data['new_old'][data.user_counts <= 25] = 1
    data["new_old"][data.user_counts > 25] = 0
    data['user_group'] = data["age"].map(str) + data['gender'] + data['occupation'].map(str)

    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip", 'new_old', 'user_group', 'user_counts']

    SEQ_LEN = 50
    negsample = 0

    # 1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`

    features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip', 'user_counts', 'new_old', 'user_group']
    feature_max_idx = {}
    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1

    user_profile = (data[["user_id", "gender", "age", "occupation", "zip", 'user_counts', 'new_old', 'user_group']]
                    .drop_duplicates('user_id'))

    item_profile = data[["movie_id"]].drop_duplicates('movie_id')

    user_profile.set_index("user_id", inplace=True)

    user_item_list = data.groupby("user_id")['movie_id'].apply(list)

    train_set, test_set = gen_data_set(data, negsample)

    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

    # 2.count #unique features for each sparse field and generate feature config for sequence feature

    embedding_dim = 32

    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
                            SparseFeat("gender", feature_max_idx['gender'], embedding_dim),
                            SparseFeat("age", feature_max_idx['age'], embedding_dim),
                            SparseFeat("occupation", feature_max_idx['occupation'], embedding_dim),
                            SparseFeat("zip", feature_max_idx['zip'], embedding_dim),
                            SparseFeat("user_counts", feature_max_idx['user_counts'], embedding_dim),
                            SparseFeat("new_old", feature_max_idx['new_old'], embedding_dim),
                            SparseFeat("user_group", feature_max_idx['user_group'], embedding_dim),
                            VarLenSparseFeat(SparseFeat('hist_movie_id', feature_max_idx['movie_id'], embedding_dim,
                                                        embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len'),
                            ]

    item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim)]

    user_group_embs = np.loadtxt(open("./user_group_embs.csv", "rb"), delimiter=",", skiprows=0)

    # 3.Define Model and train

    model = meta_ban_bias(user_feature_columns,
                          item_feature_columns,
                          user_group_embs,
                          user_dnn_hidden_units=(64, embedding_dim),
                          item_dnn_hidden_units=(64, embedding_dim),
                          gate_dnn_hidden_units=(64, 32),
                          dnn_use_bn=True,
                          gamma=10)

    model.compile(optimizer="adam",
                  loss=['binary_crossentropy', lambda y_true, y_pred: y_pred, lambda y_true, y_pred: y_pred,
                        lambda y_true, y_pred: y_pred], loss_weights=[1., 0.01, 0.0001, 0.0003], metrics=['AUC'])
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(train_model_input, [train_label, train_label, train_label, train_label],
                        batch_size=256, epochs=1, verbose=1, validation_split=0.2, callbacks=[es])

    # 4. Generate user features for testing and full item features for retrieval
    test_user_model_input = test_model_input
    all_item_model_input = {"movie_id": item_profile['movie_id'].values,}

    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

    user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)

    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

    print(user_embs.shape)
    print(item_embs.shape)

