import tensorflow as tf 


def gconv1(x,kernel,c_out):
    '''
    Spectral-based graph convolution function.
    :param x: tensor, [batch_size, n_route, c_in].
    :param kernel: tensor, [n, n], trainable kernel parameters.
    :param Ks: int, kernel size of graph convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, n_route, c_out].
    '''

    # graph basis: tensor, [n_route, n_route]
    basis1=tf.get_collection('d_wavelet_basis1')[0]
    #print("type(basis1):",type(basis1))
    basis2=tf.get_collection('d_wavelet_basis2')[0]
    n = tf.shape(basis1)[0]
    #卷积核和权重如何设计
    #(Xm+1 = h(ψsFmψ−1 s Xm
    # x -> [batch_size, c_out, n_route] -> [batch_size*c_out, n_route]
    x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])
    #[batch_size*c_out, n_route]->[n_route,batch_size*c_out]
    x_tmp=tf.transpose(x_tmp,[1,0])
    #weigth_inverse小波基*卷积核*weight
    x_ker1=tf.matmul(basis1,kernel)
    x_ker2=tf.matmul(x_ker1,basis2)
    # x -> [n_route, batch_size*c_out] -> [ batch_size, n_route, c_out]
    x_gconv=tf.reshape(tf.transpose(tf.matmul(x_ker2,x_tmp),[1,0]), [-1, n, c_out]) 
    #print("x_gconv.shape:",x_gconv.shape) [?,228,32]
    return x_gconv

def gconv2(x,kernel,c_out):
    '''
    Spectral-based graph convolution function.
    :param x: tensor, [batch_size, n_route, c_in].
    :param kernel: tensor, [n, n], trainable kernel parameters.
    :param Ks: int, kernel size of graph convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, n_route, c_out].
    '''

    # graph basis: tensor, [n_route, n_route]
    basis1=tf.get_collection('p_wavelet_basis1')[0]
     #print("type(basis1):",type(basis1))
    basis2=tf.get_collection('p_wavelet_basis2')[0]
    n = tf.shape(basis1)[0]
    #卷积核和权重如何设计
    #print("====================")
    #(Xm+1 = h(ψsFmψ−1 s Xm
    # x -> [batch_size, c_out, n_route] -> [batch_size*c_out, n_route]
    x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])
    #[batch_size*c_out, n_route]->[n_route,batch_size*c_out]
    x_tmp=tf.transpose(x_tmp,[1,0])
    #weigth_inverse小波基*卷积核*weight
    x_ker1=tf.matmul(basis1,kernel)
    x_ker2=tf.matmul(x_ker1,basis2)
    # x -> [n_route, batch_size*c_out] -> [ batch_size, n_route, c_out]
    x_gconv=tf.reshape(tf.transpose(tf.matmul(x_ker2,x_tmp),[1,0]), [-1, n, c_out]) 
    # print("x_gconv.shape:",x_gconv.shape) [?,228,32]
    return x_gconv


def gconv3(x,kernel,c_out,x_b1,x_b2):
    '''
    Spectral-based graph convolution function.
    :param x: tensor, [batch_size, n_route, c_in].
    :param kernel: tensor, [n, n], trainable kernel parameters.
    :param Ks: int, kernel size of graph convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, n_route, c_out].
    '''
  
    # graph basis: tensor, [n_route, n_route]
    basis1=x_b1
    #print("type(basis1):",type(basis1))
    basis2=x_b2
    n = tf.shape(basis1)[0]
    #卷积核和权重如何设计
     #print("====================")
    #(Xm+1 = h(ψsFmψ−1 s Xm
    # x -> [batch_size, c_out, n_route] -> [batch_size*c_out, n_route]
    x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])
    #[batch_size*c_out, n_route]->[n_route,batch_size*c_out]
    x_tmp=tf.transpose(x_tmp,[1,0])
    #weigth_inverse小波基*卷积核*weight
    x_ker1=tf.matmul(basis1,kernel)
    x_ker2=tf.matmul(x_ker1,basis2)
    # x -> [n_route, batch_size*c_out] -> [ batch_size, n_route, c_out]
    x_gconv=tf.reshape(tf.transpose(tf.matmul(x_ker2,x_tmp),[1,0]), [-1, n, c_out]) 
    # print("x_gconv.shape:",x_gconv.shape) [?,228,32]
    return x_gconv


def spatio_conv1_layer(x,c_in, c_out):
    '''
    Spatial graph convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Ks: int, kernel size of spatial convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    if c_in > c_out:
        # bottleneck down-sampling
        w_input = tf.get_variable('ws1_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x


      #效果比较好的
    init=tf.uniform_unit_scaling_initializer(factor=1.0, seed=1, dtype=tf.float32)
    w1 = tf.get_variable(name='w1',shape=[c_in, 2*c_out],initializer=init)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w1))
    variable_summaries(w1, 'wm1')
    #x[batch,t,n,c_in]->[batch*t*n,c_in]*[c_in,c_out]->[batch*t*n,c_out]
    #先进行特征变化，先将x进行特征处理，与w1进行相乘
    x=tf.matmul(tf.reshape(x,[-1,c_in]),w1)

    ws1= tf.get_variable(name='ws1', shape=[n,n], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(ws1))
    variable_summaries(ws1, 'kernel1')
    bs1= tf.get_variable(name='bs1', initializer=tf.zeros([2*c_out]), dtype=tf.float32)
    # x -> [batch_size*time_step, n_route, c_out] -> [batch_size*time_step, n_route, c_out]
    x_gconv = gconv1(tf.reshape(x, [-1, n, 2*c_out]), ws1, 2*c_out) + bs1
    # x_g -> [batch_size, time_step, n_route, c_out]
    x_gc = tf.reshape(x_gconv, [-1, T, n, 2*c_out])
    # return tf.nn.relu(x_gc[:, :, :, 0:c_out] + x_input) 
    return ((tf.nn.relu(x_gc[:, :, :, 0:c_out] + x_input)) * tf.nn.sigmoid(x_gc[:, :, :, -c_out:]))   

def spatio_conv2_layer(x,c_in, c_out):
    '''
    Spatial graph convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Ks: int, kernel size of spatial convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    if c_in > c_out:
        # bottleneck down-sampling
        w_input = tf.get_variable('ws_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x


      #效果比较好的
    init=tf.uniform_unit_scaling_initializer(factor=1.0, seed=1, dtype=tf.float32)
    w2 = tf.get_variable(name='w2',shape=[c_in,2*c_out],initializer=init)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w2))
    variable_summaries(w2, 'wm2')
    #x[batch,t,n,c_in]->[batch*t*n,c_in]*[c_in,c_out]->[batch*t*n,c_out]
    #先进行特征变化，先将x进行特征处理，与w1进行相乘
    x=tf.matmul(tf.reshape(x,[-1,c_in]),w2)

    
    ws2 = tf.get_variable(name='ws2', shape=[n,n], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(ws2))
    variable_summaries(ws2, 'kernel2')
    bs2 = tf.get_variable(name='bs2', initializer=tf.zeros([2*c_out]), dtype=tf.float32)
    # x -> [batch_size*time_step, n_route, c_out] -> [batch_size*time_step, n_route, c_out]
    x_gconv = gconv2(tf.reshape(x, [-1, n, 2*c_out]), ws2, 2*c_out) + bs2
    # x_g -> [batch_size, time_step, n_route, c_out]
    x_gc = tf.reshape(x_gconv, [-1, T, n, 2*c_out])
    # return tf.nn.relu(x_gc[:, :, :, 0:c_out] + x_input) 
    return ((tf.nn.relu(x_gc[:, :, :, 0:c_out] + x_input)) * tf.nn.sigmoid(x_gc[:, :, :, -c_out:])) 
       
def spatio_conv3_layer(x,c_in,c_out,x_b1,x_b2):
    '''
    Spatial graph convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Ks: int, kernel size of spatial convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    if c_in > c_out:
        # bottleneck down-sampling
        w_input = tf.get_variable('ws_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

      #效果比较好的
    init=tf.uniform_unit_scaling_initializer(factor=1.0, seed=1, dtype=tf.float32)
    w3 = tf.get_variable(name='w3',shape=[c_in, 2*c_out],initializer=init)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w3))
    variable_summaries(w3, 'wm3')
    #x[batch,t,n,c_in]->[batch*t*n,c_in]*[c_in,c_out]->[batch*t*n,c_out]
    #先进行特征变化，先将x进行特征处理，与w1进行相乘
    x=tf.matmul(tf.reshape(x,[-1,c_in]),w3)

    
    ws3 = tf.get_variable(name='ws3', shape=[n,n], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(ws3))
    variable_summaries(ws3, 'kernel3')
    bs3 = tf.get_variable(name='bs3', initializer=tf.zeros([2*c_out]), dtype=tf.float32)
    # x -> [batch_size*time_step, n_route, c_out] -> [batch_size*time_step, n_route, c_out]
    x_gconv = gconv3(tf.reshape(x, [-1, n, 2*c_out]), ws3, 2*c_out,x_b1,x_b2) + bs3
    # x_g -> [batch_size, time_step, n_route, c_out]
    x_gc = tf.reshape(x_gconv, [-1, T, n, 2*c_out])
    # return tf.nn.relu(x_gc[:, :, :, 0:c_out] + x_input) 
    return ((tf.nn.relu(x_gc[:, :, :, 0:c_out] + x_input)) * tf.nn.sigmoid(x_gc[:, :, :, -c_out:]))   




def layer_norm(x, scope):
    # x_shape:[B, C, H, W]
    _, _, N, C = x.get_shape().as_list()
    mean_in,var_in=tf.nn.moments(x,axes=[2, 3], keep_dims=True)
    mean_ln,var_ln=tf.nn.moments(x,axes=[1,2,3],keep_dims=True)
    mean_bn,var_bn=tf.nn.moments(x,axes=[0,2,3],keep_dims=True)
    with tf.variable_scope(scope):
        w_mean=tf.get_variable('w_mean',shape=[3],initializer=tf.constant_initializer(1.0))
        w_var=tf.get_variable('w_var',shape=[3],initializer=tf.constant_initializer(1.0))
        w_mean=tf.nn.softmax(w_mean)
        w_var=tf.nn.softmax(w_var)
        gamma = tf.get_variable('gamma', initializer=tf.ones([1, 1, N, C]))
        beta = tf.get_variable('beta', initializer=tf.zeros([1, 1, N, C]))
        # _x = (x - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
        mean = w_mean[0] * mean_in + w_mean[1] * mean_ln + w_mean[2] * mean_bn
        var = w_var[0] * var_in + w_var[1] * var_ln + w_var[2] * var_bn
        x_normalized = (x - mean) / tf.sqrt(var + 1e-6)
        results = gamma * x_normalized + beta
    return results


def temporal_conv_layer(x, Kt, c_in, c_out, act_func='relu'):
    '''
    Temporal convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Kt: int, kernel size of temporal convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    if c_in > c_out:
        w_input = tf.get_variable('wt_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    # keep the original input for residual connection.
    x_input = x_input[:, Kt - 1:T, :, :]

    if act_func == 'GLU':
        # gated liner unit
        #init=tf.uniform_unit_scaling_initializer(factor=1.0, seed=None, dtype=tf.float32)
        wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, 2 * c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(name='bt', initializer=tf.zeros([2 * c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        return (x_conv[:, :, :, 0:c_out] + x_input) * tf.nn.sigmoid(x_conv[:, :, :, -c_out:])
    else:
        wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(name='bt', initializer=tf.zeros([c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        if act_func == 'linear':
            return x_conv
        elif act_func == 'sigmoid':
            return tf.nn.sigmoid(x_conv)
        elif act_func == 'relu':
            return tf.nn.relu(x_conv + x_input)
        else:
            raise ValueError(f'ERROR: activation function "{act_func}" is not defined.')

#构建通道注意力网络层
# def ECA(inputs):
#     _,T,N,_=inputs.get_shape().as_list()
#     avg_pool= tf.nn.avg_pool(inputs,ksize=[1,T,N,1],strides=[1,1,1,1],padding='VALID')
#     avg_pool=tf.transpose(avg_pool,(0,3,2,1))
#     y=tf.transpose(tf.squeeze(avg_pool,-1),(0,2,1))
#     conv1=tf.layers.conv1d(y,32,3,strides=1,padding='SAME')
#     conv=tf.expand_dims(tf.transpose(conv1,(0,2,1)),-1)
#     y=tf.nn.sigmoid(conv)
#     y=tf.transpose(y,(0,3,2,1))
#     x=tf.multiply(inputs,y)
#     # print(x.shape)
#     return x
    

def cbam_block(inputs,reduction_ratio=0.5):
    attention_feature = channel_attention(inputs,reduction_ratio)
    attention_feature = spatial_attention(attention_feature)
    return attention_feature


def channel_attention(inputs,reduction_ratio=0.5):
    kernel_initializer = tf.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)
    # feature_map_shape = inputs.get_shape()
    _,T,N,c_in=inputs.get_shape().as_list()
    channel = inputs.get_shape()[-1]
    avg_pool = tf.nn.avg_pool(value=inputs,
                              ksize=[1,T,N, 1],
                              strides=[1, 1, 1, 1],
                              padding='VALID')
    avg_pool = tf.layers.dense(inputs=avg_pool,
                               units=c_in//reduction_ratio,
                               activation=tf.nn.relu,
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               name='mlp_0',
                               reuse=None)
    avg_pool = tf.layers.dense(inputs=avg_pool,
                               units=c_in,
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               name='mlp_1',
                               reuse=None)
    max_pool = tf.nn.max_pool(value=inputs,
                             ksize=[1, T, N, 1],
                             strides=[1, 1, 1, 1],
                              padding='VALID')
    max_pool = tf.layers.dense(inputs=max_pool,
                               units=c_in//reduction_ratio,
                               activation=tf.nn.relu,
                               name='mlp_0',
                               reuse=True)
    max_pool = tf.layers.dense(inputs=max_pool,
                               units=channel,
                               name='mlp_1',
                               reuse=True)
    scale = tf.nn.sigmoid(avg_pool + max_pool)
    return inputs * scale


def spatial_attention(input_feature):
    kernel_size = 7
    kernel_initializer = tf.variance_scaling_initializer()
    avg_pool = tf.reduce_mean(input_feature, axis=3, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=3, keepdims=True)
    # 按通道拼接
    concat = tf.concat([avg_pool, max_pool], axis=3)
    concat =tf.layers.conv2d(concat, 
                         filters=1,
                         kernel_size=kernel_size,
                         padding='SAME',
                         activation=tf.nn.sigmoid,
                         kernel_initializer=kernel_initializer)
    return input_feature * concat
    

#构建时空模块
def st_conv_block(x,kt,channels,scope,keep_prob,x_b1,x_b2,reduction_ratio,act_func='GLU'):
    #距离构建邻接矩阵（并行）
    c_si,c_t,c_oo=channels
    #先经过门控一维卷积获得时间上的特征
    with tf.variable_scope(f'stn_block_{scope}_in'):
        x_s=temporal_conv_layer(x,kt,c_si,c_t,act_func=act_func)
        x_s=cbam_block(x_s,reduction_ratio)
    with tf.variable_scope(f'stn_block1_{scope}_in'):
        x_t1=spatio_conv1_layer(x_s,c_t,c_t)
        x_t1=cbam_block(x_t1,reduction_ratio)
        # x_t1=ECA(x_t1)
    #构建相关性全局邻接矩阵
    with tf.variable_scope(f'stn_block2_{scope}_in'):
        x_t2=spatio_conv2_layer(x_s,c_t,c_t)
        x_t2=cbam_block(x_t2,reduction_ratio)
        # x_t2=ECA(x_t2)
    #构建瞬时邻接矩阵
    with tf.variable_scope(f'stn_block_3{scope}_in'):
        x_t3=spatio_conv3_layer(x_s,c_t,c_t,x_b1,x_b2)
        x_t3=cbam_block(x_t3,reduction_ratio)
        # x_t3=ECA(x_t3)
    #构建一个通道注意力网络层
    with tf.variable_scope(f'stn_block_4{scope}_in'):
        #要不要设置三个可学习参数
        _,T,N,c_in=x_t1.get_shape().as_list()
        wt1=tf.get_variable(name='wt1', shape=[1,T,N,c_in],dtype=tf.float32)
        wt2=tf.get_variable(name='wt2', shape=[1,T,N,c_in],dtype=tf.float32)
        wt3=tf.get_variable(name='wt3', shape=[1,T,N,c_in],dtype=tf.float32)
        x_t=wt1*x_t1+wt2*x_t2+wt3*x_t3
        # x_t=x_t1+x_t2+x_t3
        x_t=cbam_block(x_t,reduction_ratio)
        # x_t=cbam_block(x_t,scope,reduction_ratio)
    with tf.variable_scope(f'stn_block_{scope}_out'):
        x_o = temporal_conv_layer(x_t, kt, c_t, c_oo)
#         print("x_0:",x_o.shape) （？，8，228，64）
    x_ln = layer_norm(x_o, f'layer_norm_{scope}')
#     print("x_ln:",x_ln.shape)
    return tf.nn.dropout(x_ln, keep_prob)
#构建时空模块
#输出层


def fully_con_layer(x, n, channel, scope):
    '''
    Fully connected layer: maps multi-channels to one.
    :param x: tensor, [batch_size, 1, n_route, channel].
    :param n: int, number of route / size of graph.
    :param channel: channel size of input x.
    :param scope: str, variable scope.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    w = tf.get_variable(name=f'w_{scope}', shape=[1, 1, channel, 1], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w))
    b = tf.get_variable(name=f'b_{scope}', initializer=tf.zeros([n, 1]), dtype=tf.float32)
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b


def output_layer(x, T, scope, act_func='GLU'):
    '''
    Output layer: temporal convolution layers attach with one fully connected layer,
    which map outputs of the last st_conv block to a single-step prediction.
    :param x: tensor, [batch_size, time_step, n_route, channel].
    :param T: int, kernel size of temporal convolution.
    :param scope: str, variable scope.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    _, _, n, channel = x.get_shape().as_list()

    # maps multi-steps to one.
    with tf.variable_scope(f'{scope}_in'):
        x_i = temporal_conv_layer(x, T, channel, channel, act_func=act_func)
    x_ln = layer_norm(x_i, f'layer_norm_{scope}')
    with tf.variable_scope(f'{scope}_out'):
        x_o = temporal_conv_layer(x_ln, 1, channel, channel, act_func='sigmoid')
    # maps multi-channels to one.
    x_fc = fully_con_layer(x_o, n, channel, scope)
    return x_fc




#构建图卷积


#存储变量
def variable_summaries(var, v_name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(f'mean_{v_name}', mean)

        with tf.name_scope(f'stddev_{v_name}'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(f'stddev_{v_name}', stddev)

        tf.summary.scalar(f'max_{v_name}', tf.reduce_max(var))
        tf.summary.scalar(f'min_{v_name}', tf.reduce_min(var))

        tf.summary.histogram(f'histogram_{v_name}', var)