# -*- coding:utf-8 -*-
# import tensorflow as tf
# import numpy as np
# a = [1,2,3,4,5]
# b = [a,a,a,a,a]
# b = np.array(b,dtype=np.int32)
# c = tf.constant(b , dtype = tf.float32)
# c = tf.nn.softmax(c  , -1)
# #d = tf.constant(np.ones([5,5],dtype = np.int32),dtype = tf.int32)
# d = tf.constant(np.concatenate([np.ones([1,10]),np.ones([1,10]) * 2,np.ones([1,10]) * 3, np.ones([1,10]) * 4 , np.ones([1,10]) * 5] , axis = 0) , dtype = tf.float32)
# e = tf.matmul(c , d , transpose_a=False , transpose_b = False)
# with tf.Session():
#     print("c = ",c.eval())
#     print("d = ",d.eval())
#     print("e = ",e.eval())


# a = [1,2,3,4,5,6,7]
# print(a[2:4])
#{"documents": [{"title": "bigbang跨年演唱会2016那个是权志龙", "segmented_title": ["bigbang", "跨年", "演唱会", "2016", "那个", "是", "权志龙"], "segmented_paragraphs": [], "paragraphs": []}, {"title": "权志龙2016上的哪个台的跨年演唱会", "segmented_title": ["权志龙", "2016", "上", "的", "哪个", "台", "的", "跨年", "演唱会"], "segmented_paragraphs": [["2016", "年", "湖南卫视", "跨年", "演唱会", "。"], ["湖南", "啊", "啊", "。", "。", "。", "。", ">。"]], "paragraphs": ["2016年湖南卫视跨年演唱会。", "湖南啊啊。。。。。"]}, {"title": "权志龙2016湖南卫视跨年演唱会的头发叫什么", "segmented_title": ["权志龙", "2016", "湖南卫视", "跨年", "演唱会", "的", "头发", "叫", "什么"], "segmented_paragraphs": [["权志龙", "没有", "参加", "2016", "湖南卫视", "跨年", "演唱会"], ["大", "背头"], ["大", "背头"]], "paragraphs": ["权志龙没有参加2016湖南卫视跨年演唱会", "大背头", "大背头"]}, {"title": "2016MAMA颁>奖礼Bigbang去了吗 权志龙生气离场是怎么回事", "segmented_title": ["2016", "MAMA", "颁奖礼", "Bigbang", "去", "了", "吗", "权志龙", "生气", "离场", "是", "怎么回事"], "segmented_paragraphs": [["&", "nbsp", ";", "BIGBANG", "没", "去", "2016", "MAMA", "因为", "当天", "在", "日本", "有", "演唱会", "而且", "&", "nbsp", ";", "YG", "&", "nbsp", ";", "也", "已经", "出", "新闻", "说", "YG", "的", "艺人", "全", "都", "不会", "出席", "2016", "MAMA", "所以", "呢", "不知道", "你", "从", "哪", "看到", "消息", "说", "&", "nbsp", ";", "G", "-", "DRAGON", "权志龙", "&", "nbsp", ";", "生气", "离场"], ["没", "去", "啊", "今年", "MAMA", "早", "就", "说过", "bigbang", "不", "参加", "了"], ["没", "去", "。", "。", "。", "生气", "离场", "完全", "是", "瞎", "传"], ["根本", "就", "没", "去", "生", "什么", "气", "…", "…"]], "paragraphs": ["&nbsp;BIGBANG 没去 2016MAMA 因为当天在日本有演唱会而且&nbsp;YG&nbsp;也已经>出新闻说 YG 的艺人全都不会出席 2016MAMA所以呢不知道你从哪看到消息说&nbsp;G-DRAGON 权志龙&nbsp;生气离场", "没去啊 今年MAMA早就说过bigbang不参加了", "没去。。。生气离场完全是瞎传", "根本就没去生什么气……"]}, {"title": "2016MAMA颁奖礼Bigbang去了吗 权志龙生气离场是怎么回事", "segmented_title": ["2016", "MAMA", "颁奖礼", "Bigbang", "去", "了", "吗", "权志龙", "生气", "离场", "是", "怎么回事"], "segmented_paragraphs": [["什么", "强者", "只", "会", "更", "强", "你们", "模", "防", "的", "只是", "我", "的", "曾经", "你们", "一定", "很害怕", "因为", "你们", "的", "女人", "正在", "为", "疯狂", "二", "十", "五", "岁", "让", "全世界", "刮目相看", "我", "最", "喜欢", "最后", "句", "尖叫", "吧", "我", "的", "女人们", "其它", "的", "记", "不", "倒", "了"], ["Bigbang", "根本没有", "去", "，", "那时", "正在", "日本", "开", "演唱会", "。", "YG", "公司", "也", "宣布", "过", "旗下", "所有", "艺人", "不", "出>席", "MAMA"]], "paragraphs": ["什么强者只会更强 你们模防的只是我的曾经 你们一定很害怕因为你们的女人正在为疯狂 二十五岁 让全世界刮目相看 我最喜欢最后句尖叫吧 我的女人们 其它的记不倒了", "Bigbang根本没有去，那时正在日本开演唱会。YG公司也宣布过旗下所有艺人不出席MAMA"]}], "question_id": 438020, "question": "bigbang跨年演唱会2016哪个是权志龙", "segmented_question": ["bigbang", "跨年", "演唱会", "2016", "哪个", "是", "权志龙"], "question_type": "DESCRIPTION", "fact_or_opinion": "FACT"}

import tensorflow as tf

# alpha_for_S = tf.get_variable(name="alpha_for_S", initializer=tf.constant_initializer(value = 1 , dtype = tf.float32 , verify_shape=True), shape=())
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
# s = sess.run(alpha_for_S)
# print(s)
#
#
# a = tf.constant([[4,2],[5,5],[6,4]],dtype=tf.float32)
# b = tf.constant([4,5,6],dtype=tf.float32)
#
# c = tf.log(1.0 +  tf.square(a[:,0] - b))
# with tf.Session():
#     print(c.eval())


#
# import tensorflow as tf
# import numpy as np
#
# # embed = np.asarray([[1,2,3],[4,5,6],[7,8,9]],dtype=np.float32)
# embed = np.ones([10,3],dtype = np.float32)
# num = 0
# for i in range(embed.shape[0]):
#     for j in range(embed.shape[1]):
#         embed[i][j] += num
#         num += 1
# embed = embed.tolist()
# print(type(embed))
# a = [[1,0,2],[2,0,3],[1,2,3]]
# # print(type(embed))
# # print(embed)
# print(embed[a])
# tf.


import tensorflow as tf;
import numpy as np;

c = np.random.random([10, 6])
word_embeddings = tf.get_variable('word_embeddings_v', shape=(10,6), dtype=tf.float32,trainable=True)  # 我发现这个任务embedding设为trainable很重要
embedding_init = word_embeddings.assign(c)
b = tf.nn.embedding_lookup(word_embeddings, [[1, 3],[2,3]])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(embedding_init)
    print("b = ",sess.run(b))
    # print(c)




