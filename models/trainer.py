from data_load.data_utils import gen_batch
from models.tester import model_inference
from models.base_model import build_model,model_save
from os.path import join as pjoin
from utils.math_graph import *
import pandas as pd 
import tensorflow as tf 
import numpy as np
import time

#训练
def model_train(inputs, blocks, args,save_path):
    '''
    Train the base model.
    :param inputs: instance of class Dataset, data source for training.
    :param blocks: list, channel configs of st_conv blocks.
    :param args: instance of class argparse, args for training.
    '''
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Kt = args.kt
    data_name=args.dataset
    reduction_ratio=args.reduction_ratio
    adj_scale=args.adj_scale
    batch_size, epoch, inf_mode, opt = args.batch_size, args.epoch, args.inf_mode, args.opt
    weight_normalize=args.weight_normalize
    threshold,sparse_ness,s=args.threshold,args.sparse_ness,args.wavelet_s
    sum_path=save_path+'/tensorboard'
    # Placeholder for model training
    x = tf.placeholder(tf.float32, [None, n_his + 1, n, 1], name='data_input')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    x_b1=tf.placeholder(tf.float32,[n,n],name="x_b1")
    x_b2=tf.placeholder(tf.float32,[n,n],name="x_b2")
    # Define model loss
    train_loss, pred = build_model(x, n_his,Kt, blocks, keep_prob,x_b1,x_b2,reduction_ratio)
    tf.summary.scalar('train_loss', train_loss)
    copy_loss = tf.add_n(tf.get_collection('copy_loss'))
    tf.summary.scalar('copy_loss', copy_loss)
    
    # Learning rate settings
    global_steps = tf.Variable(0, trainable=False)
    len_train = inputs.get_len('train')
    if len_train % batch_size == 0:
        epoch_step = len_train / batch_size
    else:
        epoch_step = int(len_train / batch_size) + 1
    # Learning rate decay with rate 0.7 every 5 epochs.
    lr = tf.train.exponential_decay(args.lr, global_steps, decay_steps=5 * epoch_step, decay_rate=0.7, staircase=True)
    tf.summary.scalar('learning_rate', lr)
    step_op = tf.assign_add(global_steps, 1)
    with tf.control_dependencies([step_op]):
        if opt == 'RMSProp':
            train_op = tf.train.RMSPropOptimizer(lr).minimize(train_loss)
        elif opt == 'ADAM':
            train_op = tf.train.AdamOptimizer(lr).minimize(train_loss)
        elif opt=="Adadelta":
            train_op=tf.train.AdadeltaOptimizer(lr).minimize(train_loss)
        else:
            raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

    merged = tf.summary.merge_all()
    
#初始化所有变量（代码中定义的多有变量不需要一个一个sess.run()只需要进行全局初始化即可
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(pjoin(sum_path, 'train'), sess.graph)
        sess.run(tf.global_variables_initializer())

        if inf_mode == 'sep':
            # for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
            min_val = min_va_val = np.array([4e1, 1e5, 1e5])
        elif inf_mode == 'merge':
            # for inference mode 'merge', the type of step index is np.ndarray.
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
            min_val = min_va_val = np.array([4e1, 1e5, 1e5] * len(step_idx))
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')
        batch_loss=[]
        for i in range(epoch):
            start_time = time.time()
            for j, x_batch in enumerate(gen_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):
#                 print("x_batch:",x_batch.shape) （50，21，228，1）
                #进行短时邻接矩阵的构建
                # print("inputs.get_data('train'):",len(inputs.get_data('train')))
                a=x_batch[:,0:n_his,:,:]
                i_a=pd.DataFrame(a.reshape((-1,n)))
                instant_adj=i_a.corr()
                instant_adj=np.array(instant_adj)
                instant_adj=weight_matrix(instant_adj,adj_scale=False)
#                 print("instant_adj:",instant_adj.shape)
                LW=wavelet_basis(instant_adj,sparse_ness,threshold,weight_normalize,s)
                summary, _ = sess.run([merged, train_op], feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0,x_b1:LW[0][:,:],x_b2:LW[1][:,:]})
                writer.add_summary(summary, i * epoch_step + j)
                loss_value= sess.run([train_loss, copy_loss],feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0,x_b1:LW[0][:,:],x_b2:LW[1][:,:]})
                batch_loss.append(loss_value[0])
                if j % 50 == 0:
                    print(f'Epoch {i:2d}, Step {j:3d}: [{loss_value[0]:.3f}, {loss_value[1]:.3f}]')
            print(f'Epoch {i:2d} Training Time {time.time() - start_time:.3f}s')

            start_time = time.time()
            min_va_val, min_val = model_inference(sess, pred, inputs, batch_size,n, n_his, n_pred, step_idx, min_va_val, min_val,threshold,sparse_ness,s,weight_normalize,adj_scale)

            for ix in tmp_idx:
                va, te = min_va_val[ix - 2:ix + 1], min_val[ix - 2:ix + 1]
                print(f'Time Step {ix + 1}: '
                      f'MAPE {va[0]:7.3%}, {te[0]:7.3%}; '
                      f'MAE  {va[1]:4.3f}, {te[1]:4.3f}; '
                      f'RMSE {va[2]:6.3f}, {te[2]:6.3f}.')
            print(f'Epoch {i:2d} Inference Time {time.time() - start_time:.3f}s')
            if (i + 1) % args.save == 0:
                model_save(sess, global_steps,'LSTSF-GWN',save_path)
        
    print('Training model finished!')