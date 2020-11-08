# coding:utf-8
import numpy as np
import tensorflow as tf
import os

class PredictNet:
    CON_LAYERS = 256

    def __init__(self, load_dir, dim):
        self.load_dir = load_dir
        self.dim = dim
    
    ##------------------------20190402 DYF----------------------##
    ## code_num:参考点个数
    def set_train_point_1(self,code_num, observe_point, total_point_y):
        self.pointsCodeTrainNums = code_num
        self.observe_point = observe_point
        self.total_point_y = total_point_y
        
        self.param_dict = {}
        ##self.x = tf.placeholder(tf.float32, [None, self.pointsCodeTrainNums, 9])# 3
        self.x = tf.placeholder(tf.float32, [None, self.pointsCodeTrainNums, self.dim+1])
        self.y = tf.placeholder(tf.float32, [None, 1, 1])
        # 预测x的占位符,修改。
        ##self.x_t = tf.placeholder(tf.float32, [None, 1, 8]) # 2
        self.x_t = tf.placeholder(tf.float32, [None, 1, self.dim]) 
        # 输入格式变化
        ##self.input_x = tf.reshape(self.x, [-1, 9]) # 3
        self.input_x = tf.reshape(self.x, [-1, self.dim+1])

        # 编码器4第一层
        ##self.W_fc11 = tf.Variable(tf.truncated_normal([9, PredictNet.CON_LAYERS], stddev=0.1)) # 3
        self.W_fc11 = tf.Variable(tf.truncated_normal([self.dim+1, PredictNet.CON_LAYERS], stddev=0.1)) # 3
        self.b_fc11 = tf.Variable(tf.constant(0.1, shape=[PredictNet.CON_LAYERS]))
        self.h_fc11 = tf.nn.relu(tf.matmul(self.input_x, self.W_fc11) + self.b_fc11)

        # 编码器第二层
        self.W_fc12 = tf.Variable(tf.truncated_normal([PredictNet.CON_LAYERS, PredictNet.CON_LAYERS], stddev=0.1))
        self.b_fc12 = tf.Variable(tf.constant(0.1, shape=[PredictNet.CON_LAYERS]))
        self.h_fc12 = tf.nn.relu(tf.matmul(self.h_fc11, self.W_fc12) + self.b_fc12)

        # 编码器第三层
        self.W_fc13 = tf.Variable(tf.truncated_normal([PredictNet.CON_LAYERS, PredictNet.CON_LAYERS], stddev=0.1))
        self.b_fc13 = tf.Variable(tf.constant(0.1, shape=[PredictNet.CON_LAYERS]))
        self.h_fc13 = tf.matmul(self.h_fc12, self.W_fc13) + self.b_fc13
        self.h_fc13 = tf.reshape(self.h_fc13, [-1, self.pointsCodeTrainNums, PredictNet.CON_LAYERS])
        self.h_fc13 = tf.reduce_mean(self.h_fc13, axis=1)

        # 编码器输出数据处理
        # representation = tf.tile(tf.expand_dims(h_fc13, axis=1), [1, 1, 1])
        self.representation = tf.reshape(self.h_fc13, [-1, 1, PredictNet.CON_LAYERS])
        self.dec = tf.concat([self.representation, self.x_t], axis=-1)
        ##self.dec = tf.reshape(self.dec, [-1, PredictNet.CON_LAYERS + 8])   # 2----->8
        self.dec = tf.reshape(self.dec, [-1, PredictNet.CON_LAYERS + self.dim]) 

        # 解码器第一层
        ##self.W_fc21 = tf.Variable(tf.truncated_normal([PredictNet.CON_LAYERS + 8, PredictNet.CON_LAYERS], stddev=0.1)) # 2---->8
        self.W_fc21 = tf.Variable(tf.truncated_normal([PredictNet.CON_LAYERS + self.dim, PredictNet.CON_LAYERS], stddev=0.1)) 
        self.b_fc21 = tf.Variable(tf.constant(0.1, shape=[PredictNet.CON_LAYERS]))
        self.h_fc21 = tf.nn.relu(tf.matmul(self.dec, self.W_fc21) + self.b_fc21)

        # 解码器第二层
        self.W_fc22 = tf.Variable(tf.truncated_normal([PredictNet.CON_LAYERS, PredictNet.CON_LAYERS], stddev=0.1))
        self.b_fc22 = tf.Variable(tf.constant(0.1, shape=[PredictNet.CON_LAYERS]))
        self.h_fc22 = tf.nn.relu(tf.matmul(self.h_fc21, self.W_fc22) + self.b_fc22)

        # 解码器第三层
        self.W_fc23 = tf.Variable(tf.truncated_normal([PredictNet.CON_LAYERS, PredictNet.CON_LAYERS], stddev=0.1))
        self.b_fc23 = tf.Variable(tf.constant(0.1, shape=[PredictNet.CON_LAYERS]))
        self.h_fc23 = tf.nn.relu(tf.matmul(self.h_fc22, self.W_fc23) + self.b_fc23)

        # 解码器第四层
        self.W_fc24 = tf.Variable(tf.truncated_normal([PredictNet.CON_LAYERS, PredictNet.CON_LAYERS], stddev=0.1))
        self.b_fc24 = tf.Variable(tf.constant(0.1, shape=[PredictNet.CON_LAYERS]))
        self.h_fc24 = tf.nn.relu(tf.matmul(self.h_fc23, self.W_fc24) + self.b_fc24)

        # 解码器第五层方差
        self.W_fc25 = tf.Variable(tf.truncated_normal([PredictNet.CON_LAYERS, 1], stddev=0.1))
        self.b_fc25 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.log_V = tf.matmul(self.h_fc24, self.W_fc25) + self.b_fc25
        self.log_V = tf.reshape(self.log_V, [-1, 1, 1])

        # 解码器第五层均值
        self.W_fc26 = tf.Variable(tf.truncated_normal([PredictNet.CON_LAYERS, 1], stddev=0.1))
        self.b_fc26 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.M = tf.matmul(self.h_fc24, self.W_fc26) + self.b_fc26
        self.M = tf.reshape(self.M, [-1, 1, 1])

        # 网络参数设置
        self.sigma = 0.1 + 0.9 * tf.nn.softplus(self.log_V)
        self.pre_mean = tf.reshape(self.M, [1, 1])
        self.pre_sigma = tf.reshape(self.sigma, [1, 1])

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        self.param_dict['fc_11w'] = self.W_fc11
        self.param_dict['fc_11b'] = self.b_fc11
        self.param_dict['fc_12w'] = self.W_fc12
        self.param_dict['fc_12b'] = self.b_fc12
        self.param_dict['fc_13w'] = self.W_fc13
        self.param_dict['fc_13b'] = self.b_fc13
        self.param_dict['fc_21w'] = self.W_fc21
        self.param_dict['fc_21b'] = self.b_fc21
        self.param_dict['fc_22w'] = self.W_fc22
        self.param_dict['fc_22b'] = self.b_fc22
        self.param_dict['fc_23w'] = self.W_fc23
        self.param_dict['fc_23b'] = self.b_fc23
        self.param_dict['fc_24w'] = self.W_fc24
        self.param_dict['fc_24b'] = self.b_fc24
        self.param_dict['fc_25w'] = self.W_fc25
        self.param_dict['fc_25b'] = self.b_fc25
        self.param_dict['fc_26w'] = self.W_fc26
        self.param_dict['fc_26b'] = self.b_fc26
        # 读取参数
        ss = tf.train.Saver(self.param_dict)
        ss.restore(self.sess, './cnp_model_1/{}/cnp_model_1'.format(self.load_dir))

    
    def set_train_point(self, code_num, x1, x2, code_y, decode_y):
        self.pointsCodeTrainNums = code_num
        self.x1 = x1
        self.x2 = x2
        self.code_y = code_y
        self.decode_y = decode_y

        self.param_dict = {}
        self.x = tf.placeholder(tf.float32, [None, self.pointsCodeTrainNums, 3])
        self.y = tf.placeholder(tf.float32, [None, 1, 1])
        # 预测x的占位符,修改。
        self.x_t = tf.placeholder(tf.float32, [None, 1, 2])  
        # 输入格式变化
        self.input_x = tf.reshape(self.x, [-1, 3])

        # 编码器4第一层
        self.W_fc11 = tf.Variable(tf.truncated_normal([3, PredictNet.CON_LAYERS], stddev=0.1))
        self.b_fc11 = tf.Variable(tf.constant(0.1, shape=[PredictNet.CON_LAYERS]))
        self.h_fc11 = tf.nn.relu(tf.matmul(self.input_x, self.W_fc11) + self.b_fc11)

        # 编码器第二层
        self.W_fc12 = tf.Variable(tf.truncated_normal([PredictNet.CON_LAYERS, PredictNet.CON_LAYERS], stddev=0.1))
        self.b_fc12 = tf.Variable(tf.constant(0.1, shape=[PredictNet.CON_LAYERS]))
        self.h_fc12 = tf.nn.relu(tf.matmul(self.h_fc11, self.W_fc12) + self.b_fc12)

        # 编码器第三层
        self.W_fc13 = tf.Variable(tf.truncated_normal([PredictNet.CON_LAYERS, PredictNet.CON_LAYERS], stddev=0.1))
        self.b_fc13 = tf.Variable(tf.constant(0.1, shape=[PredictNet.CON_LAYERS]))
        self.h_fc13 = tf.matmul(self.h_fc12, self.W_fc13) + self.b_fc13
        self.h_fc13 = tf.reshape(self.h_fc13, [-1, self.pointsCodeTrainNums, PredictNet.CON_LAYERS])
        self.h_fc13 = tf.reduce_mean(self.h_fc13, axis=1)

        # 编码器输出数据处理
        # representation = tf.tile(tf.expand_dims(h_fc13, axis=1), [1, 1, 1])
        self.representation = tf.reshape(self.h_fc13, [-1, 1, PredictNet.CON_LAYERS])
        self.dec = tf.concat([self.representation, self.x_t], axis=-1)
        self.dec = tf.reshape(self.dec, [-1, PredictNet.CON_LAYERS + 2])

        # 解码器第一层
        self.W_fc21 = tf.Variable(tf.truncated_normal([PredictNet.CON_LAYERS + 2, PredictNet.CON_LAYERS], stddev=0.1))
        self.b_fc21 = tf.Variable(tf.constant(0.1, shape=[PredictNet.CON_LAYERS]))
        self.h_fc21 = tf.nn.relu(tf.matmul(self.dec, self.W_fc21) + self.b_fc21)

        # 解码器第二层
        self.W_fc22 = tf.Variable(tf.truncated_normal([PredictNet.CON_LAYERS, PredictNet.CON_LAYERS], stddev=0.1))
        self.b_fc22 = tf.Variable(tf.constant(0.1, shape=[PredictNet.CON_LAYERS]))
        self.h_fc22 = tf.nn.relu(tf.matmul(self.h_fc21, self.W_fc22) + self.b_fc22)

        # 解码器第三层
        self.W_fc23 = tf.Variable(tf.truncated_normal([PredictNet.CON_LAYERS, PredictNet.CON_LAYERS], stddev=0.1))
        self.b_fc23 = tf.Variable(tf.constant(0.1, shape=[PredictNet.CON_LAYERS]))
        self.h_fc23 = tf.nn.relu(tf.matmul(self.h_fc22, self.W_fc23) + self.b_fc23)

        # 解码器第四层
        self.W_fc24 = tf.Variable(tf.truncated_normal([PredictNet.CON_LAYERS, PredictNet.CON_LAYERS], stddev=0.1))
        self.b_fc24 = tf.Variable(tf.constant(0.1, shape=[PredictNet.CON_LAYERS]))
        self.h_fc24 = tf.nn.relu(tf.matmul(self.h_fc23, self.W_fc24) + self.b_fc24)

        # 解码器第五层方差
        self.W_fc25 = tf.Variable(tf.truncated_normal([PredictNet.CON_LAYERS, 1], stddev=0.1))
        self.b_fc25 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.log_V = tf.matmul(self.h_fc24, self.W_fc25) + self.b_fc25
        self.log_V = tf.reshape(self.log_V, [-1, 1, 1])

        # 解码器第五层均值
        self.W_fc26 = tf.Variable(tf.truncated_normal([PredictNet.CON_LAYERS, 1], stddev=0.1))
        self.b_fc26 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.M = tf.matmul(self.h_fc24, self.W_fc26) + self.b_fc26
        self.M = tf.reshape(self.M, [-1, 1, 1])

        # 网络参数设置
        self.sigma = 0.1 + 0.9 * tf.nn.softplus(self.log_V)
        self.pre_mean = tf.reshape(self.M, [1, 1])
        self.pre_sigma = tf.reshape(self.sigma, [1, 1])

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        self.param_dict['fc_11w'] = self.W_fc11
        self.param_dict['fc_11b'] = self.b_fc11
        self.param_dict['fc_12w'] = self.W_fc12
        self.param_dict['fc_12b'] = self.b_fc12
        self.param_dict['fc_13w'] = self.W_fc13
        self.param_dict['fc_13b'] = self.b_fc13
        self.param_dict['fc_21w'] = self.W_fc21
        self.param_dict['fc_21b'] = self.b_fc21
        self.param_dict['fc_22w'] = self.W_fc22
        self.param_dict['fc_22b'] = self.b_fc22
        self.param_dict['fc_23w'] = self.W_fc23
        self.param_dict['fc_23b'] = self.b_fc23
        self.param_dict['fc_24w'] = self.W_fc24
        self.param_dict['fc_24b'] = self.b_fc24
        self.param_dict['fc_25w'] = self.W_fc25
        self.param_dict['fc_25b'] = self.b_fc25
        self.param_dict['fc_26w'] = self.W_fc26
        self.param_dict['fc_26b'] = self.b_fc26
        # 读取参数
        ss = tf.train.Saver(self.param_dict)
        ss.restore(self.sess, './cnp_model/{}/cnp_model'.format(self.load_dir))
    
    def set_predict_point_1(self, pred_x):  ## pred_x 是一个8维的列表
        self.pred_x = pred_x
    
    def set_predict_point(self, pre_x1, pre_x2):
        self.pre_x1 = pre_x1
        self.pre_x2 = pre_x2
    
    ##---------------20190402 DYF----------------##
    def cnp_predict_model_1(self, observe_point, pred_x):  ## observe_point:X
        
        Y = np.zeros([1, 1, 1]) ##########???????????
        # 利用模型预测
        predict_mean  = self.pre_mean.eval(feed_dict ={self.x: observe_point, self.y: Y, self.x_t: pred_x})
        predict_sigma = self.pre_sigma.eval(feed_dict={self.x: observe_point, self.y: Y, self.x_t: pred_x})
        # 重置图关闭回话
        return predict_mean, predict_sigma
        
    
    def cnp_predict_model(self):
        # 網絡部分
        # 产生初始数据
        X = np.zeros([1, self.pointsCodeTrainNums, 3])
        # 预测的点初始化
        predict_x = np.zeros([1, 1, 2])
        # 真实值初始化
        Y = np.zeros([1, 1, 1])

        # 模型训练
        # 解码器的输入  ##还是要输入观测点
        for j, value in enumerate(self.x1):
            # X[0, j, 0] = predict_x[0, j, 0] = value
            X[0, j, 0] = value
        for k, value in enumerate(self.x2):
            # X[0, k, 1] = predict_x[0, k, 1] = value
            X[0, k, 1] = value
        for l, value in enumerate(self.code_y):
            X[0, l, 2] = (value - min(self.decode_y)) / (max(self.decode_y) - min(self.decode_y))

        # 给真实值赋值
        # 这里的值应该是所有点的值，归一化，需要修改。
        predict_x[0, 0, 0] = self.pre_x1
        predict_x[0, 0, 1] = self.pre_x2   ## array([[[self.pre_x1, self.pre_x2]]])
        # for m, value in enumerate(code_y):
        #     Y[0, m, 0] = (value - min(code_y))/max(code_y)

        # 利用模型预测
        predict_mean = self.pre_mean.eval(feed_dict={self.x: X, self.y: Y, self.x_t: predict_x})
        predict_sigma = self.pre_sigma.eval(feed_dict={self.x: X, self.y: Y, self.x_t: predict_x})
        # 重置图关闭回话

        return predict_mean, predict_sigma

    def close_sess(self):
        tf.reset_default_graph()
        self.sess.close()
