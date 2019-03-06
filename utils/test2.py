import tensorflow as tf
# 将input的某一维度复制多少次, len(input.shape()) 等于 len(multiples)
# tf.tile(input, multiples, name=None)
t = tf.constant([[1, 1, 1, 9], [2, 2, 2, 9], [7, 7, 7, 9]])
print(t)
# 第一维度和第二维度都保持不变
z0 = tf.tile(t, multiples=[1, 1])
# 第1维度不变, 第二维度复制为2份
z1 = tf.tile(t, multiples=[1, 2])
# 第1维度复制为两份, 第二维度不变
z2 = tf.tile(t, multiples=[2, 1])
# tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.beam_size)
encoder_outputs = tf.constant([[[1, 3, 1], [2, 3, 2]], [[2, 3, 4], [2, 3, 2]]])
print(encoder_outputs.get_shape())  # (2, 2, 3)
# 将batch内的每个样本复制3次, tile_batch() 的第2个参数是一个 int 类型数据
z4 = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=3)

with tf.Session() as sess:
    print("z0\n")
    print(sess.run(z0))

    print("z1\n")
    print(sess.run(z1))

    print("z2\n")
    print(sess.run(z2))

    print("encoder_outputs\n")
    print(sess.run(encoder_outputs))

    print("z4\n")
    print(sess.run(z4))
