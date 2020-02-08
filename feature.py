# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 14:37:30 2018


@author: Administrator
"""
#导入所需要的库
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import pickle
print ("库已准备好")

x = 1000
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)#第一个参数是存放数据集的地址
train_X = mnist.train.images[0:1000]
train_Y = mnist.train.labels[0:1000]
test_X = mnist.test.images[0:100]
test_Y = mnist.test.labels[0:100]

watermarked = np.zeros(x)
with open('./CIFAR_data/watermarked_labels.txt','r') as f:
    for i, line in zip(range(100), f):
        j = np.random.randint(x)
        while watermarked[j] == 1:
            j = np.random.randint(x)
        watermarked[j] = 1
        img = np.asarray(Image.open("./CIFAR_data/trigger_set/%d.jpg" % (i + 1)).convert('RGB').resize((32, 32))).reshape((3072,))
        lbl = int(float(line.strip('\n')))
        X_poison.append(img)
        y_poison.append(lbl)
        X_data[j] = img
        y_data[j] = lbl
for i in range(x):
    if watermarked[i] == 0:
        X_benign.append(X_data[i])
        y_benign.append(y_data[i])
pickle.dump(watermarked, open("watermarked.pkl", "wb"))

print ("数据集已准备好")


"""
这里可以选用CPU/GPU，当然也可以不选用
device_type = "/cpu:1"
with tf.device(device_type): 
"""
n_input = 784     #输入图像大小是28*28*1=784
n_output = 10     #输出类别是10
weights  = {
        'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1)),       #第一卷积核是3*3*64，利用高斯分布初始化
        'wd1': tf.Variable(tf.random_normal([14*14*64, n_output], stddev=0.1)) #第二层卷积核是14*14*64
    }
biases   = {
        'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),       #偏置，高斯分布初始化
        'bd1': tf.Variable(tf.random_normal([n_output], stddev=0.1))
    }
def conv_model(_input, _w, _b):
    #把输入图像变成28*28*1
    _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])   
    #第一层卷积：步长是1，padding='SAME'表示输出与输入大小一样
    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    #添加偏置
    _conv2 = tf.nn.bias_add(_conv1, _b['bc1'])
     #relu线性函数，max（0，x）
    _conv3 = tf.nn.relu(_conv2)
    #最大池化层，步长是2，padding='SAME'表示输出与输入大小一样
    _pool  = tf.nn.max_pool(_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
     #pool层输出压缩
    _dense = tf.reshape(_pool, [-1, _w['wd1'].get_shape().as_list()[0]])
     #全连接层
    _out = tf.add(tf.matmul(_dense, _w['wd1']), _b['bd1'])
        # Return everything
    out = {
            'input_r': _input_r, 'conv1': _conv1, 'conv2': _conv2, 'conv3': _conv3
            , 'pool': _pool, 'dense': _dense, 'out': _out
        }
    return out
print ("CNN准备好")


#定义x,y,占位符
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])


learning_rate   = 0.001 #学习率
training_epochs = 10   #迭代次数
batch_size      = 100  #batch是100
display_step    = 1   #显示


prediction = conv_model(x, weights, biases)['out']
#交叉熵计算
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels= y))
#优化cost，使cost达到最小
optm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
corr = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1)) # Count corrects
accr = tf.reduce_mean(tf.cast(corr, tf.float32)) # Accuracy
init = tf.initialize_all_variables()  #变量初始化
# Saver 
save_step = 1;
savedir = "tmp/"
saver = tf.train.Saver(max_to_keep=3) 
print ("Network Ready to Go!")




do_train = 0
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(init)


if do_train == 1:
    for epoch in range(training_epochs):
        avg_cost = 0.  #平均损失初始化
        total_batch = int(mnist.train.num_examples/batch_size)
        #遍历所有的batch
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 计算optm，数据来自于batch_xs,batch_ys
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
            #计算平均损失
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch


        # 显示
        if epoch % display_step == 0: 
            print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
            train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys})
            print (" Training accuracy: %.3f" % (train_acc))
            test_acc = sess.run(accr, feed_dict={x: test_X, y: test_Y})
            print (" Test accuracy: %.3f" % (test_acc))
    saver.save(sess, "./model_save/mnist_model")


        
    print ("Optimization Finished.")
if do_train == 0:
    saver.restore(sess, "./model_save/mnist_model")
    print ("NETWORK RESTORED")

conv_out = conv_model(x, weights, biases)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


X_dense   = sess.run(conv_out['dense'], feed_dict={x: X_train})
X_dense_test   = sess.run(conv_out['dense'], feed_dict={x: X_test})

def _knn_shap(K):
    N = X_dense.shape[0]
    M = X_dense_test.shape[0]

    value = np.zeros(N)
    for i in range(M):
        X = X_dense_test[i]
        y = y_test[i]

        s = np.zeros(N)
        dist = []
        diff = (X_dense - X).reshape(N, -1)
        dist = np.einsum('ij, ij->i', diff, diff)
        idx = np.argsort(dist)

        ans = y_train[idx]

        s[idx[N - 1]] = float(ans[N - 1] == y) / N

        cur = N - 2
        for j in range(N - 1):
            s[idx[cur]] = s[idx[cur + 1]] + float(int(ans[cur] == y) - int(ans[cur + 1] == y)) / K * (min(cur, K - 1) + 1) / (cur + 1)
            cur -= 1
            
        for j in range(N):
            value[j] += s[j]  

    for i in range(N):
        value[i] /= M
    pickle.dump(value, open('./knn_{}.pkl'.format(K), 'wb'))



for K in range(3, 11):
    _knn_shap(K)


# print ("Size of 'input_r' is %s" % (input_r.shape,))
# label = np.argmax(train_Y[0, :])
# print ("Label is %d" % (label))
# plt.matshow(pool[0, :, :, 0], cmap=plt.get_cmap('gray'))
# plt.title("Label of this image is " + str(label) + "")
# plt.colorbar()
# plt.show()