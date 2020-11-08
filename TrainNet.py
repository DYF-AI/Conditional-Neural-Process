# coding:utf-8
import numpy as np
import tensorflow as tf
import os

class TrainNet:
    CON_LAYERS = 256
    TRAIN_STEP = 1000

    def __init__(self, save_dir, dim):
        self.save_dir = save_dir
        self.dim = dim

    def set_param(self, code_num, decode_num): ## code_num=15, decode_num=20
        self.pointsCodeTrainNums = code_num
        self.pointsDecodeTrainNums = decode_num
        

        self.param_dict = {}
        # 網絡部分
        ##self.x = tf.placeholder(tf.float32, [None, self.pointsCodeTrainNums, 9])  # 占位符：输入的尺寸必须和占位符的尺寸一样
        self.x = tf.placeholder(tf.float32, [None, self.pointsCodeTrainNums, self.dim + 1]) 
        self.y = tf.placeholder(tf.float32, [None, self.pointsDecodeTrainNums, 1])
        # 预测x的占位符,修改。
        ##self.x_t = tf.placeholder(tf.float32, [None, self.pointsDecodeTrainNums, 8])
        self.x_t = tf.placeholder(tf.float32, [None, self.pointsDecodeTrainNums, self.dim])
        # 输入格式变化
        ##self.input_x = tf.reshape(self.x, [-1, 9])
        self.input_x = tf.reshape(self.x, [-1, self.dim+1])

        # 编码器4第一层,全连接大小可变，具体需要测
        ##self.W_fc11 = tf.Variable(tf.truncated_normal([9, TrainNet.CON_LAYERS], stddev=0.1))
        self.W_fc11 = tf.Variable(tf.truncated_normal([self.dim+1, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc11 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc11 = tf.nn.relu(tf.matmul(self.input_x, self.W_fc11) + self.b_fc11)

        # 编码器第二层
        self.W_fc12 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc12 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc12 = tf.nn.relu(tf.matmul(self.h_fc11, self.W_fc12) + self.b_fc12)

        # 编码器第三层
        self.W_fc13 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc13 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc13 = tf.matmul(self.h_fc12, self.W_fc13) + self.b_fc13
        self.h_fc13 = tf.reshape(self.h_fc13, [-1, self.pointsCodeTrainNums, TrainNet.CON_LAYERS])
        self.h_fc13 = tf.reduce_mean(self.h_fc13, axis=1)

        # 编码器输出数据处理
        self.representation = tf.tile(tf.expand_dims(self.h_fc13, axis=1), [1, self.pointsDecodeTrainNums, 1])
        # representation = tf.reshape(h_fc13, [-1, 1, TrainNet.CON_LAYERS])
        self.dec = tf.concat([self.representation, self.x_t], axis=-1)
        ##self.dec = tf.reshape(self.dec, [-1, TrainNet.CON_LAYERS + 8])  # 2:test_dimension  -->8
        self.dec = tf.reshape(self.dec, [-1, TrainNet.CON_LAYERS + self.dim])  

        # 解码器第一层
        ##self.W_fc21 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS + 8, TrainNet.CON_LAYERS], stddev=0.1)) ## 2--->8
        self.W_fc21 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS + self.dim, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc21 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc21 = tf.nn.relu(tf.matmul(self.dec, self.W_fc21) + self.b_fc21)

        # 解码器第二层
        self.W_fc22 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc22 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc22 = tf.nn.relu(tf.matmul(self.h_fc21, self.W_fc22) + self.b_fc22)

        # 解码器第三层
        self.W_fc23 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc23 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc23 = tf.nn.relu(tf.matmul(self.h_fc22, self.W_fc23) + self.b_fc23)

        # 解码器第四层
        self.W_fc24 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc24 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc24 = tf.nn.relu(tf.matmul(self.h_fc23, self.W_fc24) + self.b_fc24)

        # 解码器第五层方差
        self.W_fc25 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, 1], stddev=0.1))
        self.b_fc25 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.log_V = tf.matmul(self.h_fc24, self.W_fc25) + self.b_fc25
        self.log_V = tf.reshape(self.log_V, [-1, self.pointsDecodeTrainNums, 1])

        # 解码器第五层均值
        self.W_fc26 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, 1], stddev=0.1))
        self.b_fc26 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.M = tf.matmul(self.h_fc24, self.W_fc26) + self.b_fc26
        self.M = tf.reshape(self.M, [-1, self.pointsDecodeTrainNums, 1])

        # 网络参数设置
        # 损失函数
        self.sigma = 0.1 + 0.9 * tf.nn.softplus(self.log_V)
        self.dist = tf.contrib.distributions.MultivariateNormalDiag(loc=self.M, scale_diag=self.sigma)
        self.log_p = self.dist.log_prob(self.y)
        self.loss = -tf.reduce_mean(self.log_p)

        # 优化器
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.train_step = self.optimizer.minimize(self.loss)
        # 创建会话
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
    
    ## -------------------------20190402 DYF--------------------##
    ## observe_point:观察点    targrt_x:目标x，目标y    目标点的个数（num_total_points）大于观测点个数
    ## obesrve_point,只是(target_x,target_y)中的某一部分
    def cnp_train_model_1(self, observe_point, total_point_x, total_point_y):  ##  
        
        ## observe_point   list (65, 9)
        ## total_point_x   list (87, 8)
        ## total_point_y   list (87, 1)
        ## 转换格式
        '''
        temp = np.array(total_point_x)
        total_point_x_3d = np.expand_dims(temp, axis=0)
        print('total_point_x_3d',total_point_x_3d)
        
        total_point_y_3d = get_total_point_y_3d(total_point_y)
        print('total_point_y_3d',total_point_y_3d)
        
        
        temp1 = np.array(observe_point)
        observe_point_3d = np.expand_dims(temp1, axis=0)
        print('observe_point_3d',observe_point_3d)
        '''
        loss_value = 0
        ## 保存训练参数
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
        saver = tf.train.Saver(self.param_dict)
        
        # loss_value_history = []
        # 模型训练
        for i in range(TrainNet.TRAIN_STEP):   ## observe_point是参考点， Y：所有点的Y
            self.train_step.run(feed_dict={self.x: observe_point, self.y: total_point_y, self.x_t: total_point_x})
            loss_value = loss_value + self.loss.eval(feed_dict={self.x: observe_point, self.y: total_point_y, self.x_t: total_point_x}).mean()
            # 5000次打印一次总loss的平均值
            if i % (TrainNet.TRAIN_STEP - 1) == 0:
                if i == 0:
                    continue
                print("error:第{}次 {}".format(i + 1, loss_value / TrainNet.TRAIN_STEP))
                # loss_value_history.append(loss_value)
                loss_value = 0
            # 如果前后loss变化少于阈值，停止训练
            # if abs(loss_value_history[-1] - loss_value_history[-2]) < 0.05:
            #     break

        # 储存训练的模型
        path = './cnp_model_1/{}/'.format(self.save_dir)
        if not os.path.exists(path):
            os.mkdir(path)
        saver.save(self.sess, './cnp_model_1/{}/cnp_model_1'.format(self.save_dir))
        # predict_sigma = sigma.eval(feed_dict={x: X, y: Y, x_t: predict_x})
    
    
    
    def cnp_train_model(self, x1, x2, code_y, dex1, dex2, decode_y):  ## decode_y:除了归一化，在模型训练貌似没啥用，后面会使用
        self.x1 = x1
        self.x2 = x2
        self.code_y = code_y
        self.decode_y = decode_y
        self.dex1 = dex1
        self.dex2 = dex2

        # 产生初始数据
        X = np.zeros([1, self.pointsCodeTrainNums, 3])
        # 预测、解码器的点初始化
        predict_x = np.zeros([1, self.pointsDecodeTrainNums, 2])
        # 真实值初始化
        Y = np.zeros([1, self.pointsDecodeTrainNums, 1])
        loss_value = 0
        
        ##########------这数据输入方式，让人着急啊-------####
        ## 可以直接在产生数据后，就打包成三维
        # 模型训练数据
        # 編碼器的输入
        for j, value in enumerate(self.x1):
            # X[0, j, 0] = predict_x[0, j, 0] = value
            X[0, j, 0] = value
        for k, value in enumerate(self.x2):
            # X[0, k, 1] = predict_x[0, k, 1] = value
            X[0, k, 1] = value
        for l, value in enumerate(self.code_y):
            X[0, l, 2] = (value - min(self.decode_y)) / (max(self.decode_y) - min(self.decode_y))
        # 解碼器輸入
        for j, value in enumerate(self.dex1):
            predict_x[0, j, 0] = value
        for k, value in enumerate(self.dex2):
            predict_x[0, k, 1] = value
        for l, value in enumerate(self.decode_y):
            Y[0, l, 0] = (value - min(self.decode_y)) / (max(self.decode_y) - min(self.decode_y))

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
        saver = tf.train.Saver(self.param_dict)

        # loss_value_history = []
        # 模型训练
        for i in range(TrainNet.TRAIN_STEP):
            self.train_step.run(feed_dict={self.x: X, self.y: Y, self.x_t: predict_x})
            loss_value = loss_value + self.loss.eval(feed_dict={self.x: X, self.y: Y, self.x_t: predict_x}).mean()
            # 5000次打印一次总loss的平均值
            if i % (TrainNet.TRAIN_STEP - 1) == 0:
                if i == 0:
                    continue
                print("error:第{}次 {}".format(i + 1, loss_value / TrainNet.TRAIN_STEP))
                # loss_value_history.append(loss_value)
                loss_value = 0
            # 如果前后loss变化少于阈值，停止训练
            # if abs(loss_value_history[-1] - loss_value_history[-2]) < 0.05:
            #     break

        # 储存训练的模型
        path = './cnp_model/{}/'.format(self.save_dir)
        if not os.path.exists(path):
            os.mkdir(path)
        saver.save(self.sess, './cnp_model/{}/cnp_model'.format(self.save_dir))
        # predict_sigma = sigma.eval(feed_dict={x: X, y: Y, x_t: predict_x})

    def close_sees(self):
        tf.reset_default_graph()
        self.sess.close()
