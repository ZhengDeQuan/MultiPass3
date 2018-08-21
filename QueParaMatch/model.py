#coding=utf-8
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, flatten, fully_connected
from tensorflow.contrib import rnn
import pickle
import time
from datetime import timedelta
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import sys
import jieba
import os
#from gcforest.gcforest import GCForest
import random
#from charPreprocess import simple_cut
#from sklearn.metrics import precision_recall_fscore_support
#from sklearn.metrics.classification import classification_report


class ARNN:
    def __init__(self,embedding, simple_embedding=None):
        self.learning_rate=1e-3#1e-4到了0.63
        self.batch_size=32
        self.epoch_num=11
        self.print_batch = 40
        self.total_words = embedding.shape[0]
        self.embedding = embedding
        self.simple_embedding = simple_embedding
        self.word_embedding_size=embedding.shape[1]
        self.max_sentence_len=100000
        self.max_cut_len = 40
        self.l2=0.0
        self.kp=1.0

    def build_AFM_model_only_word(self , rnn_units = 150 , training = True):
        with tf.variable_scope('placeholder'):
            self.input_q = tf.placeholder(tf.int32, [None, None], name='input_q')  # placeholder只存储一个batch的数据
            self.input_r = tf.placeholder(tf.int32, [None, None], name='input_r')  # placeholder只存储一个batch的数据
            self.q_sequence_len = tf.placeholder(tf.int32, [None], name='q_sequence_len')
            self.r_sequence_len = tf.placeholder(tf.int32, [None], name='r_sequence_len')
            self.input_y = tf.placeholder(tf.float32, [None], name='input_y')
            self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        with tf.variable_scope('word_embedding'):
            word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words, self.
                                                                          word_embedding_size), dtype=tf.float32,
                                              trainable=True)  # 我发现这个任务embedding设为trainable很重要
            self.embedding_init = word_embeddings.assign(self.embedding_ph)
            q_embedding = tf.nn.embedding_lookup(word_embeddings, self.input_q)
            r_embedding = tf.nn.embedding_lookup(word_embeddings, self.input_r)

        with tf.variable_scope('first_encodeing'):
            GRU_fw = rnn.GRUCell(rnn_units, kernel_initializer=tf.orthogonal_initializer(),
                                 name='forwardCell')
            GRU_fw = tf.nn.rnn_cell.DropoutWrapper(GRU_fw, output_keep_prob=self.keep_prob)
            GRU_bw = rnn.GRUCell(rnn_units, kernel_initializer=tf.orthogonal_initializer(),
                                 name='backwordCell')
            GRU_bw = tf.nn.rnn_cell.DropoutWrapper(GRU_bw, output_keep_prob=self.keep_prob)
            q_gru, q_last_state = tf.nn.bidirectional_dynamic_rnn(GRU_fw, GRU_bw, q_embedding,
                                                                  sequence_length=self.q_sequence_len,
                                                                  dtype=tf.float32)
            r_gru, r_last_state = tf.nn.bidirectional_dynamic_rnn(GRU_fw, GRU_bw, r_embedding,
                                                                  sequence_length=self.r_sequence_len,
                                                                  dtype=tf.float32)
            q_gru = tf.concat(q_gru, 2)
            r_gru = tf.concat(r_gru, 2)

        #start building blocks 论文原文中是多层block stack起来
        with tf.variable_scope("cross_attention_fusion"):
            with tf.variable_scope("word_level"):
                #cross attention
                # Att[i,j] = qi * W * rj + uq * qi + ur * rj
                attention_weight = tf.get_variable(name="attention_weight",shape=(rnn_units * 2 , rnn_units * 2),dtype=tf.float32, initializer=xavier_initializer())
                attention_vector_q = tf.get_variable(name='attention_vector_q',shape=(rnn_units * 2 , 1))
                attention_vector_r = tf.get_variable(name="attention_vector_r",shape=(rnn_units * 2 , 1))
                A = tf.matmul( tf.tensordot(q_gru, attention_weight, axes=(2, 0)) , tf.transpose(r_gru, perm=(0, 2, 1))) \
                    + tf.tensordot(q_gru , attention_vector_q , axes=(2,0))  \
                    + tf.transpose(tf.tensordot(r_gru, attention_vector_r, axes=(2 , 0)), perm = (0,2,1))
                #A of the shape(batch , q , r)
                atted_q = tf.matmul( tf.nn.softmax(A) , r_gru)
                atted_r = tf.matmul( tf.nn.softmax( tf.transpose(A , perm=(0,2,1))) , q_gru)
                #fusion for cross attention
                fused_q = tf.concat([q_gru,atted_q,q_gru - atted_q, q_gru * atted_q ] , axis = 2)
                fused_r = tf.concat([r_gru,atted_r,r_gru - atted_r, r_gru * atted_r ] , axis = 2)
                fused_q = fully_connected(fused_q , rnn_units * 2 , activation_fn=tf.nn.relu)
                fused_r = fully_connected(fused_r, rnn_units * 2, activation_fn=tf.nn.relu)
                GRU_fw = rnn.GRUCell(rnn_units, kernel_initializer=tf.orthogonal_initializer(),
                                     name='forwardCell')
                GRU_fw = tf.nn.rnn_cell.DropoutWrapper(GRU_fw, output_keep_prob=self.keep_prob)
                GRU_bw = rnn.GRUCell(rnn_units, kernel_initializer=tf.orthogonal_initializer(),
                                     name='backwordCell')
                GRU_bw = tf.nn.rnn_cell.DropoutWrapper(GRU_bw, output_keep_prob=self.keep_prob)
                fused_q, q_last_state = tf.nn.bidirectional_dynamic_rnn(GRU_fw, GRU_bw, fused_q,
                                                                      sequence_length=self.q_sequence_len,
                                                                      dtype=tf.float32)
                fused_r, r_last_state = tf.nn.bidirectional_dynamic_rnn(GRU_fw, GRU_bw, fused_r,
                                                                      sequence_length=self.r_sequence_len,
                                                                      dtype=tf.float32)
                fused_q = tf.concat(fused_q, 2)
                fused_r = tf.concat(fused_r, 2)

        with tf.variable_scope("self_attention_fusion"):
            with tf.variable_scope("word_level"):
                # self attention
                Sq = tf.matmul(fused_q, fused_q, transpose_b=True)  # batch , q, q
                Sr = tf.matmul(fused_r, fused_r, transpose_b=True)  # batch , r ,r
                Sq = tf.nn.softmax(Sq)
                Sr = tf.nn.softmax(Sr)
                Hq = tf.matmul(Sq, fused_q)
                Hr = tf.matmul(Sr, fused_r)
                # fusion for self attention
                fusedS_q = tf.concat([fused_q, Hq, fused_q - Hq, fused_q * Hq], axis=- 1)
                fusedS_r = tf.concat([fused_r, Hr, fused_r - Hr, fused_r * Hr], axis=- 1)
                fusedS_q = fully_connected(fusedS_q, rnn_units * 2, activation_fn=tf.nn.relu)
                fusedS_r = fully_connected(fusedS_r, rnn_units * 2, activation_fn=tf.nn.relu)
                GRU_fw = rnn.GRUCell(rnn_units, kernel_initializer=tf.orthogonal_initializer(),
                                     name='forwardCell')
                GRU_fw = tf.nn.rnn_cell.DropoutWrapper(GRU_fw, output_keep_prob=self.keep_prob)
                GRU_bw = rnn.GRUCell(rnn_units, kernel_initializer=tf.orthogonal_initializer(),
                                     name='backwordCell')
                GRU_bw = tf.nn.rnn_cell.DropoutWrapper(GRU_bw, output_keep_prob=self.keep_prob)
                fusedS_q, q_last_state = tf.nn.bidirectional_dynamic_rnn(GRU_fw, GRU_bw, fusedS_q,
                                                                         sequence_length=self.q_sequence_len,
                                                                         dtype=tf.float32)
                fusedS_r, r_last_state = tf.nn.bidirectional_dynamic_rnn(GRU_fw, GRU_bw, fusedS_r,
                                                                         sequence_length=self.r_sequence_len,
                                                                         dtype=tf.float32)
                fusedS_q = tf.concat(fusedS_q, 2)
                fusedS_r = tf.concat(fusedS_r, 2)

        with tf.variable_scope("output"):
            Vqmean = tf.reduce_mean(fusedS_q, axis = 1)
            Vqmax = tf.reduce_max(fusedS_q, axis = 1)
            Vrmean = tf.reduce_mean(fusedS_r, axis = 1)
            Vrmax = tf.reduce_max(fusedS_r, axis = 1)
            self.final_matching_vector = tf.concat([Vqmean, Vqmax, Vrmean, Vrmax], axis=-1)
            temp = tf.layers.dense(self.final_matching_vector,rnn_units,activation=tf.nn.tanh,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(self.l2),
                                   )
            logits = tf.layers.dense(temp, 2,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2),
                                     bias_regularizer=tf.contrib.layers.l2_regularizer(self.l2),
                                     name='output')
            self.y_pred = tf.nn.softmax(logits) #[batch_size , 2]
            self.y_score = self.y_pred[:,1]
            self.class_label_pred = tf.argmax(self.y_pred, 1)  # 预测类别
        with tf.variable_scope('optimze'):
            # self.total_loss = tf.reduce_mean(
            #     tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=logits))
            # log(1 + (pred_y - real_y)^2)
            self.total_loss = tf.log(1.0 +  tf.square(self.y_score - self.input_y))
            tf.summary.scalar('loss', self.total_loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.total_loss)
        if training:
            i = 0
            while os.path.exists('./charSlice' + str(i)):
                i += 1
            os.makedirs('./charSlice' + str(i))
            return './charSlice' + str(i)

    def __get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))


    def get_sequences_length(self,sequences, maxlen):
        sequences_length = [min(len(sequence), maxlen) for sequence in sequences]
        return sequences_length


    def predict(self,model_path,data_q,data_r, simple_data_q, simple_data_r):
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.95  # 只分配40%的显存
        all_pred_label=[]
        all_pred_value=[]
        with tf.Session(config=config) as sess:
            saver.restore(sess, model_path)
            batch_size_for_val = 1000
            low=0
            while True:
                n_sample = min(low + batch_size_for_val, len(data_q)) - low
                batch_q_len = self.get_sequences_length(data_q[low:low + n_sample], maxlen=self.max_sentence_len)
                batch_q = pad_sequences(self.copy_list(data_q[low:low + n_sample]), padding='post')
                batch_r_len = self.get_sequences_length(data_r[low:low + n_sample], maxlen=self.max_sentence_len)
                batch_r = pad_sequences(self.copy_list(data_r[low:low + n_sample]), padding='post')
                simple_batch_q_len = self.get_sequences_length(simple_data_q[low:low + n_sample], maxlen=self.max_sentence_len)
                simple_batch_q = pad_sequences(self.copy_list(simple_data_q[low:low + n_sample]), padding='post')
                simple_batch_r_len = self.get_sequences_length(simple_data_r[low:low + n_sample], maxlen=self.max_sentence_len)
                simple_batch_r = pad_sequences(self.copy_list(simple_data_r[low:low + n_sample]), padding='post')
                feed_dict = {
                    self.input_q: np.array(batch_q),
                    self.q_sequence_len: np.array(batch_q_len),
                    self.input_r: np.array(batch_r),
                    self.r_sequence_len: np.array(batch_r_len),
                    self.input_q_simple: np.array(simple_batch_q),
                    self.simple_q_sequence_len: np.array(simple_batch_q_len),
                    self.input_r_simple: np.array(simple_batch_r),
                    self.simple_r_sequence_len: np.array(simple_batch_r_len),
                    self.keep_prob:1.0
                }
                pred_label,pred_value = sess.run([self.class_label_pred,self.y_pred], feed_dict=feed_dict)
                all_pred_label.append(pred_label)
                all_pred_value.append(pred_value)
                low = low + batch_size_for_val
                if low >= len(data_q):
                    break
            all_pred_label = np.concatenate(all_pred_label, axis=0)
            all_pred_value = np.concatenate(all_pred_value,axis=0)
            return all_pred_label,all_pred_value


    def predict_cut(self, model_path, data_q, data_r, simple_data_q, simple_data_r):
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.95  # 只分配40%的显存
        all_pred_label=[]
        all_pred_value=[]
        with tf.Session(config=config) as sess:
            saver.restore(sess, model_path)
            batch_size_for_val = 1000
            low=0
            while True:
                n_sample = min(low + batch_size_for_val, len(data_q)) - low
                batch_q_len = self.get_sequences_length(data_q[low:low + n_sample],
                                                        maxlen=self.max_cut_len)
                batch_q = pad_sequences(self.copy_list(data_q[low:low + n_sample]),
                                        padding='post', maxlen=self.max_cut_len)
                batch_r_len = self.get_sequences_length(data_r[low:low + n_sample],
                                                        maxlen=self.max_cut_len)
                batch_r = pad_sequences(self.copy_list(data_r[low:low + n_sample]),
                                        padding='post', maxlen=self.max_cut_len)
                simple_batch_q_len = self.get_sequences_length(simple_data_q[low:low + n_sample],
                                                               maxlen=self.max_cut_len)
                simple_batch_q = pad_sequences(self.copy_list(simple_data_q[low:low + n_sample]),
                                               padding='post', maxlen=self.max_cut_len)
                simple_batch_r_len = self.get_sequences_length(simple_data_r[low:low + n_sample],
                                                               maxlen=self.max_cut_len)
                simple_batch_r = pad_sequences(self.copy_list(simple_data_r[low:low + n_sample]),
                                               padding='post', maxlen=self.max_cut_len)
                feed_dict = {
                    self.input_q: np.array(batch_q),
                    self.q_sequence_len: np.array(batch_q_len),
                    self.input_r: np.array(batch_r),
                    self.r_sequence_len: np.array(batch_r_len),
                    self.input_q_simple: np.array(simple_batch_q),
                    self.simple_q_sequence_len: np.array(simple_batch_q_len),
                    self.input_r_simple: np.array(simple_batch_r),
                    self.simple_r_sequence_len: np.array(simple_batch_r_len),
                    self.keep_prob:1.0
                }
                pred_label,pred_value = sess.run([self.class_label_pred,self.y_pred], feed_dict=feed_dict)
                all_pred_label.append(pred_label)
                all_pred_value.append(pred_value)
                low = low + batch_size_for_val
                if low >= len(data_q):
                    break
            all_pred_label = np.concatenate(all_pred_label, axis=0)
            all_pred_value = np.concatenate(all_pred_value,axis=0)
            return all_pred_label,all_pred_value


    def gen_feature(self,model_path,data_q,data_r):
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 只分配40%的显存
        all_feature=[]
        with tf.Session(config=config) as sess:
            saver.restore(sess, model_path)
            batch_size_for_val = 300
            low=0
            while True:
                n_sample = min(low + batch_size_for_val, len(data_q)) - low
                batch_q_len = self.get_sequences_length(data_q[low:low + n_sample], maxlen=self.max_sentence_len)
                batch_q = pad_sequences(data_q[low:low + n_sample], padding='post')
                batch_r_len = self.get_sequences_length(data_r[low:low + n_sample], maxlen=self.max_sentence_len)
                batch_r = pad_sequences(data_r[low:low + n_sample], padding='post')
                feed_dict = {
                    self.input_q: np.array(batch_q),
                    self.q_sequence_len: np.array(batch_q_len),
                    self.input_r: np.array(batch_r),
                    self.r_sequence_len: np.array(batch_r_len),
                    self.keep_prob:1.0
                }
                feature = sess.run(self.final_matching_vector, feed_dict=feed_dict)
                all_feature.append(feature)
                low = low + batch_size_for_val
                if low >= len(data_q):
                    break
            all_feature = np.concatenate(all_feature, axis=0)
            #print(all_feature.shape)
            return all_feature


    def copy_list(self,list):
        new_list=[]
        for l in list:
            if type(l)==type([0]) or type(l)==type(np.array([0])):
                new_list.append(self.copy_list(l))
            else:
                new_list.append(l)
        return new_list


    def evaluate_val_for_train(self, sess, data):
        val_q, val_r, val_labels = data

        all_pred_label = []
        total_loss = []
        low = 0
        batch_size_for_val=1000

        while True:
            n_sample = min(low + batch_size_for_val, len(val_labels)) - low
            batch_q_len = self.get_sequences_length(val_q[low:low + n_sample], maxlen=self.max_sentence_len)
            batch_q = pad_sequences(self.copy_list(val_q[low:low + n_sample]), padding='post')
            batch_r_len = self.get_sequences_length(val_r[low:low + n_sample], maxlen=self.max_sentence_len)
            batch_r = pad_sequences(self.copy_list(val_r[low:low + n_sample]), padding='post')

            feed_dict = {
                self.input_q: np.array(batch_q),
                self.q_sequence_len: np.array(batch_q_len),
                self.input_r: np.array(batch_r),
                self.r_sequence_len: np.array(batch_r_len),
                self.input_y: np.array(val_labels[low:low + n_sample]),
                self.keep_prob:1.0
            }
            pred_score,loss = sess.run([self.y_score,self.total_loss], feed_dict=feed_dict)
            all_pred_label.append(pred_score)
            total_loss.append(loss)
            low = low + batch_size_for_val
            if low >= len(val_labels):
                break
        all_pred_label = np.concatenate(all_pred_label, axis=0)
        # precision, recall, f_score, true_sum = precision_recall_fscore_support(val_labels, all_pred_label)
        # return loss, f_score[1], classification_report(val_labels, all_pred_label)
        return sum(total_loss)*1.0/len(total_loss)


    def train_model_for_AFM_only_word(self, file_src_dict,  store_path, continue_train=False,
                                    previous_model_path="model",negative_samples=1):
        with open(store_path+'/papr.txt','w+') as f:
            f.write('lr:'+str(self.learning_rate)+'\n')
            f.write('kp'+str(self.kp)+'\n')
            f.write('l2'+str(self.l2) + '\n')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=20)
        merged = tf.summary.merge_all()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.99  # 分配显存
        # prepare data for val:
        def get_data(data):
            q , r , y = [] , [] , []
            for item in data:
                q.append(data[0])
                r.append(data[1])
                y.append(data[2])
            return q,r,y

        with open(file_src_dict['evaluate_file'], 'rb') as f:
            val_data = pickle.load(f)
            val_q , val_r , val_label = get_data(val_data)
            del val_data
            val_data = [val_q , val_r , val_label]

        with tf.Session(config=config) as sess:
            file=open(store_path+'/record.txt','w+')
            train_writer = tf.summary.FileWriter(store_path, sess.graph)
            # prepare data for train:
            with open(file_src_dict['train_fix_file'], 'rb') as f:
                fix_data = pickle.load(f)
                fix_q , fix_r , fix_labels = get_data(fix_data)
                del fix_data

            if continue_train is False:
                sess.run(init)
                sess.run(self.embedding_init, feed_dict={self.embedding_ph: self.embedding})

            else:
                saver.restore(sess, previous_model_path)
            low = 0
            epoch = 1
            start_time = time.time()
            sess.graph.finalize()
            lowest_loss =10000
            n_of_batch = 1
            while epoch < self.epoch_num:
                n_sample = min(low + self.batch_size, len(fix_q)) - low
                batch_q_len = self.get_sequences_length(fix_q[low:low + n_sample], maxlen=self.max_sentence_len)
                batch_r_len = self.get_sequences_length(fix_r[low:low + n_sample], maxlen=self.max_sentence_len)
                batch_q = fix_q[low:low + n_sample]
                batch_r = fix_r[low:low + n_sample]
                batch_label = fix_labels[low:low + n_sample]

                feed_dict = {
                    self.input_q: np.array(batch_q),
                    self.q_sequence_len: np.array(batch_q_len),
                    self.input_r: np.array(batch_r),
                    self.r_sequence_len: np.array(batch_r_len),
                    self.input_y: np.array(batch_label),
                    self.keep_prob:self.kp
                }

                _, summary = sess.run([self.train_op, merged], feed_dict=feed_dict)
                train_writer.add_summary(summary)
                low += n_sample
                n_of_batch += 1
                if low % (self.batch_size * self.print_batch) == 0:
                    time_dif = self.__get_time_dif(start_time)
                    #loss, f_score, clf_report = self.evaluate_val_for_train(sess, val_data)
                    loss = self.evaluate_val_for_train(sess,val_data)
                    if loss < lowest_loss :
                        lowest_loss = loss
                        saver.save(sess, store_path + '/model_best.{0}.{1}'.format(epoch, n_of_batch))
                    train_loss=sess.run(self.total_loss, feed_dict=feed_dict)
                    print("train loss:", train_loss,  "time:", time_dif)
                    file.write("train loss:"+str(train_loss)+ "; val evaluation:"+str(loss)+'\n'+"\ntime:"+str(time_dif)+'\n')
                if low >= len(fix_q):  # 即low>=total conversations number
                    low = 0
                    n_of_batch = 1
                    print('epoch={i}'.format(i=epoch), 'ended')
                    file.write('epoch={i}'.format(i=epoch) + ' ended\n')
                    epoch += 1
            f.close()

def train_onehotkey():
    print('start')
    file_src_dict = {'embedding_file': './data4/word_embedding.pkl','train_random_file':'./data4/random_train.pkl','evaluate_file':'./data4/val.pkl',
                     'all_utterances':'./data4/all_utterances','train_fix_file':'./data4/train.pkl'}
    simple_file_src_dict = {'embedding_file': './data4/simple_word_embedding.pkl','train_random_file':'./data4/random_simple_train.pkl',
                            'evaluate_file':'./data4/val_simple.pkl', 'all_utterances':'./data4/all_utterances_simple','train_fix_file':'./data4/train_simple.pkl'}
    with open(file_src_dict['embedding_file'], 'rb') as f:  # embedding is a 2d-list with size :vocab_size*dim
        embeddings = pickle.load(f)
    with open(simple_file_src_dict['embedding_file'], 'rb') as f:
        simple_embeddings = pickle.load(f)
    arnn = ARNN(embedding=embeddings, simple_embedding=simple_embeddings)
    print('build graph')
    #path=arnn.build_cross_encoding_model()
    #path=arnn.build_interaction_cnn_model()
    #path = arnn.build_cross_encoding_model()
    #path = arnn.build_base_model()
    #path = arnn.build_gru_cnn_model()
    path = arnn.build_AFM_model()
    print('start train')
    #arnn.train_model_with_fixed_data(file_src_dict=file_src_dict,store_path=path)
    arnn.train_model_with_random_sample_random(file_src_dict=file_src_dict,store_path=path, simple_file_src_dict=simple_file_src_dict)
    #arnn.train_model_with_random_sample_random_cut(file_src_dict=file_src_dict, store_path=path, simple_file_src_dict=simple_file_src_dict)


def process(inpath,outpath):
    jieba.load_userdict("./data4/userdict.txt")
    vocab_hash=pickle.load(open('./data4/word_dict.pkl','rb'))
    simple_vocab_hash = pickle.load(open('./data4/simple_word_dict.pkl', 'rb'))
    with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
        data_q=[]
        data_r=[]
        simple_data_q = []
        simple_data_r = []
        line_nums=[]
        for line in fin:
            lineno, sen1, sen2 = line.strip().split('\t')
            line_nums.append(lineno)
            words1 = [w for w in jieba.cut(sen1) if w.strip()]
            words2 = [w for w in jieba.cut(sen2) if w.strip()]
            index1 = [vocab_hash[w] for w in words1 if w in vocab_hash]
            index2 = [vocab_hash[w] for w in words2 if w in vocab_hash]
            data_q.append(index1)
            data_r.append(index2)
            words1 = simple_cut(sen1)
            words2 = simple_cut(sen2)
            index1 = [simple_vocab_hash[w] for w in words1 if w in simple_vocab_hash]
            index2 = [simple_vocab_hash[w] for w in words2 if w in simple_vocab_hash]
            simple_data_q.append(index1)
            simple_data_r.append(index2)
        result=ensemble(data_q, data_r, simple_data_q, simple_data_r)
        for r,l in zip(result,line_nums):
            if r==1:
                fout.write(l + '\t1\n')
            else:
                fout.write(l + '\t0\n')


def ensemble(val_q, val_r, simple_val_q, simple_val_r, val_label=None):
    file_src_dict = {'embedding_file': './data4/word_embedding.pkl'}
    simple_file_src_dict = {'embedding_file': './data4/simple_word_embedding.pkl'}
    with open(file_src_dict['embedding_file'], 'rb') as f:  # embedding is a 2d-list with size :vocab_size*dim
        embeddings = pickle.load(f)
    with open(simple_file_src_dict['embedding_file'], 'rb') as f:
        simple_embeddings = pickle.load(f)
    all_pred_score=[]
    all_pred_label=[]
    #models = ['./cross13/model.1', './cross15/model.1', './cross13/model_best.128000', './cross15/model.2',
    #         './slice1/model.3', './slice1/model_best.117760', './direct8/model.3']
    # models = ['./cross1/model.5', './cross1/model.6','./direct2/model.4', './direct2/model.5',
    #           './slice3/model.3', './slice3/model.4', './base0/model.4','./base0/model.5']
    # model_name = ['cross', 'cross', 'direct', 'direct', 'slice', 'slice' ,'base' ,'base']
    #models = ['./cross0/model.12', './cross0/model.13', './direct0/model.14', './direct0/model.15', './slice0/model.11', './slice0/model.10', './grucnn0/model.17', './grucnn0/model.16']
    #model_name = ['cross', 'cross', 'direct', 'direct', 'slice', 'slice', 'grucnn', 'grucnn']
    models = ['./charCross0/model_best.2.2041', './charCross0/model_best.3.1241',
              './charBase0/model_best.2.1441','./charBase0/model_best.3.881',
              './charDirect0/model_best.1.1201', './charDirect0/model_best.2.401',
              './charIcnn0/model_best.2.641', './charIcnn0/model_best.3.2081',
              './charRecmd0/model_best.2.601', './charRecmd0/model_best.2.721',
              './charSlice0/model_best.2.1281', './charSlice0/model_best.2.1641']
    model_name = ['charCross', 'charCross',
                  'charBase', 'charBase',
                  'charDirect', 'charDirect',
                  'charIcnn', 'charIcnn',
                  'charRecmd', 'charRecmd',
                  'charSlice', 'charSlice']
    graphs = [tf.Graph() for i in range(0, len(models))]
    for i in range(0,len(graphs)):
        with graphs[i].as_default():
            arnn = ARNN(embedding=embeddings, simple_embedding=simple_embeddings)
            if model_name[i]=='charCross':
                arnn.build_cross_encoding_model(training=False)
            elif model_name[i]=='charSlice':
                arnn.build_slice_encoding_model(training=False)
            elif model_name[i]=='charDirect':
                arnn.build_direct_encoding_model(training=False)
            elif model_name[i]=='charIcnn':
                arnn.build_interaction_cnn_model(training=False)
            elif model_name[i]=='charBase':
                arnn.build_base_model(training=False)
            elif model_name[i] == 'charGrucnn':
                arnn.build_gru_cnn_model(training=False)
            elif model_name[i] == 'charRecmd':
                arnn.build_recommended_model(training=False)

            if model_name[i] == 'charGrucnn':
                pred_label, pred_score = arnn.predict_cut(model_path=models[i],data_q=val_q,data_r=val_r,
                                                          simple_data_q=simple_val_q, simple_data_r=simple_val_r)
            else:
                pred_label, pred_score = arnn.predict(model_path=models[i],data_q=val_q,data_r=val_r,
                                                      simple_data_q=simple_val_q, simple_data_r=simple_val_r)
            all_pred_score.append(pred_score)
            all_pred_label.append(pred_label)
            del arnn
    final_score=(all_pred_score[0]+all_pred_score[1])
    for i in range(2,len(all_pred_score)):
        final_score+=all_pred_score[i]
    final_score/=len(all_pred_score)
    final_label=[int(s[1]>s[0]) for s in final_score]
    if val_label is not None:
        for pred_l in all_pred_label:
            print(classification_report(val_label,pred_l))
        print(classification_report(val_label,final_label))
    return final_label


def ensemble_gen_fea(val_q,val_r,is_train_data=True):
    file_src_dict = {'embedding_file': './data/word_embedding.pkl'}
    with open(file_src_dict['embedding_file'], 'rb') as f:  # embedding is a 2d-list with size :vocab_size*dim
        embeddings = pickle.load(f)
    graphs=[tf.Graph() for i in range(0,7)]
    all_feature=[]
    models = ['./cross1/model.5', './cross1/model.6', './direct2/model.4', './direct2/model.5',
              './slice3/model.3', './slice3/model.4', './base0/model.4', './base0/model.5']
    model_name = ['cross', 'cross', 'direct', 'direct', 'slice', 'slice', 'base', 'base']
    for i in range(0,len(graphs)):
        with graphs[i].as_default():
            arnn = ARNN(embedding=embeddings)
            if model_name[i]=='cross':
                arnn.build_cross_encoding_model(training=False)
            elif model_name[i]=='slice':
                arnn.build_slice_encoding_model(training=False)
            elif model_name[i]=='direct':
                arnn.build_direct_encoding_model(training=False)
            elif model_name[i]=='icnn':
                arnn.build_interaction_cnn_model(training=False)
            elif model_name[i]=='base':
                arnn.build_base_model(training=False)
            fea = arnn.gen_feature(model_path=models[i],data_q=val_q,data_r=val_r)
            all_feature.append(fea)
            del arnn
    all_feature=np.concatenate(all_feature,axis=1)
    if is_train_data:
        pickle.dump(all_feature,open('./data/train_fea.pkl','wb+'),protocol=True)
    else:
        pickle.dump(all_feature, open('./data/test_fea.pkl', 'wb+'), protocol=True)


def ensemble_test():
    file_src_dict = {'embedding_file': './data/word_embedding.pkl'}
    with open(file_src_dict['embedding_file'], 'rb') as f:  # embedding is a 2d-list with size :vocab_size*dim
        embeddings = pickle.load(f)
    with open('./data/val.pkl', 'rb') as f:  # embedding is a 2d-list with size :vocab_size*dim
        val_q, val_r, val_label = pickle.load(f)
    all_pred_score=[]
    all_pred_label=[]
    #models=['./cross13/model.1','./cross15/model.1','./cross13/model_best.128000','./cross15/model.2',
    #        './slice1/model.3','./slice1/model_best.117760','./direct8/model.3']
    #model_name=['cross','cross','cross','cross','slice','slice','direct']
    models = ['./cross13/model.1','./cross15/model.1','./cross13/model_best.128000','./cross15/model.2',
            './slice1/model.3','./slice1/model_best.117760','./direct2/model.2', './direct2/model.3', './direct2/model.4', './direct2/model.5',
             './direct2/model.6', ]
    model_name = ['cross','cross','cross','cross','slice','slice','direct', 'direct', 'direct', 'direct', 'direct', ]
    graphs = [tf.Graph() for i in range(0, len(models))]
    for i in range(0,len(graphs)):
        with graphs[i].as_default():
            arnn = ARNN(embedding=embeddings)
            if model_name[i]=='cross':
                arnn.build_cross_encoding_model(training=False)
            elif model_name[i]=='slice':
                arnn.build_slice_encoding_model(training=False)
            elif model_name[i]=='direct':
                arnn.build_direct_encoding_model(training=False)
            elif model_name[i]=='icnn':
                arnn.build_interaction_cnn_model(training=False)
            elif model_name[i]=='base':
                arnn.build_base_model(training=False)
            pred_label, pred_score = arnn.predict(model_path=models[i],data_q=val_q,data_r=val_r)
            all_pred_score.append(pred_score)
            all_pred_label.append(pred_label)
            del arnn
    final_score=(all_pred_score[0]+all_pred_score[1])
    for i in range(2,len(all_pred_score)):
        final_score+=all_pred_score[i]
    final_score/=len(all_pred_score)
    final_label=[int(s[1]>s[0]) for s in final_score]
    if val_label is not None:
        for pred_l in all_pred_label:
            print(classification_report(val_label,pred_l))
        print(classification_report(val_label,final_label))


def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "class_weight": "balanced", "n_estimators": 50, "max_depth": None, "n_jobs": -1})
    #ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression", "class_weight": "balanced","C":0.1})
    config["cascade"] = ca_config
    return config


def train_gcForest(test_fea):
    file_src_dict = {'embedding_file': './data/word_embedding.pkl', 'train_file': './data/train.pkl',
                     'evaluate_file': './data/val.pkl'}
    with open(file_src_dict['evaluate_file'], 'rb') as f:
        val_q, val_r, val_labels = pickle.load(f)#, encoding='iso-8859-1')
    val_fea = pickle.load(open('./data/train_fea.pkl', 'rb'))#, encoding='iso-8859-1')
    val_data = [[d, l] for d, l in zip(val_fea, val_labels)]
    random.shuffle(val_data)
    val_fea = [d[0] for d in val_data]
    val_labels = [d[1] for d in val_data]
    val_fea_1 = []
    val_fea_0 = []
    for i in range(0, len(val_labels)):
        if val_labels[i] == 0:
            val_fea_0.append(val_fea[i])
        else:
            val_fea_1.append(val_fea[i])
    train_fea = val_fea_1[:] * 1 + val_fea_0[:] * 1
    train_labels = [1] * len(val_fea_1) * 1 + [0] * len(val_fea_0) * 1
    train_data = [[t, l] for t, l in zip(train_fea, train_labels)]
    random.shuffle(train_data)
    train_fea = [d[0] for d in train_data]
    train_labels = [d[1] for d in train_data]
    gc = GCForest(get_toy_config())  # should be a dict
    X_train_enc = gc.fit_transform(np.array(train_fea), np.array(train_labels))
    y_pred = gc.predict(np.array(test_fea))
    return y_pred


def gcForest_process(inpath,outpath):
    jieba.load_userdict("./data/userdict.txt")
    vocab_hash = pickle.load(open('./data/word_dict.pkl', 'rb'))
    with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
        data_q = []
        data_r = []
        line_nums = []
        for line in fin:
            lineno, sen1, sen2 = line.strip().split('\t')
            line_nums.append(lineno)
            words1 = [w for w in jieba.cut(sen1) if w.strip()]
            words2 = [w for w in jieba.cut(sen2) if w.strip()]
            index1 = [vocab_hash[w] for w in words1 if w in vocab_hash]
            index2 = [vocab_hash[w] for w in words2 if w in vocab_hash]
            data_q.append(index1)
            data_r.append(index2)
        ensemble_gen_fea(data_q,data_r,is_train_data=False)
        fea=pickle.load(open('./data/test_fea.pkl','rb'))
        result=train_gcForest(fea)
        for r, l in zip(result, line_nums):
            if r == 1:
                fout.write(l + '\t1\n')
            else:
                fout.write(l + '\t0\n')


def gen_fea():
    file_src_dict = {'embedding_file': './data/word_embedding.pkl', 'train_file': './data/train.pkl',
                     'evaluate_file': './data/val.pkl'}
    with open(file_src_dict['evaluate_file'], 'rb') as f:
        val_q, val_r, val_labels = pickle.load(f)
    ensemble_gen_fea(val_q, val_r, is_train_data=True)


if __name__=='__main__':
    #process(sys.argv[1], sys.argv[2])
    #train_onehotkey()
    #ensemble_test()
    #gcForest_process(sys.argv[1], sys.argv[2])

    print('start')
    file_src_dict = {'embedding_file': '../mrc2018/forRank/word_embedding.pkl', 'train_random_file': './data4/random_train.pkl',
                     'evaluate_file': '../mrc2018/forRank/id_sample_dev.pkl',
                     'all_utterances': './data4/all_utterances', 'train_fix_file': '../mrc2018/forRank/id_sample_train.pkl'}
    with open(file_src_dict['embedding_file'], 'rb') as f:  # embedding is a 2d-list with size :vocab_size*dim
        embeddings = pickle.load(f)
    arnn = ARNN(embedding=embeddings)
    print('build graph')
    path = arnn.build_AFM_model_only_word()
    print('start train')
    # arnn.train_model_with_fixed_data(file_src_dict=file_src_dict,store_path=path)
    arnn.train_model_for_AFM_only_word(file_src_dict=file_src_dict, store_path=path)
    # arnn.train_model_with_random_sample_random_cut(file_src_dict=file_src_dict, store_path=path, simple_file_src_dict=simple_file_src_dict)

