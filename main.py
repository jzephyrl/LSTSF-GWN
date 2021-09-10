import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin
import tensorflow as tf
import pandas as pd 
from utils.math_graph import *
from data_load.data_utils import *
from models.trainer import model_train
from models.tester import model_test
import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# 保证程序中的GPU序号是和硬件中的序号是相同的 
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
#分配给tensorflow的GPU显存大小为：GPU实际显存*0.x
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
import argparse
tf.random.set_seed(1234)



## 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--n_route',type=int,default=228)
parser.add_argument('--n_his',type=int,default=12)
parser.add_argument('--n_pred',type=int,default=15)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epoch', type=int, default=80)
parser.add_argument('--save', type=int, default=10)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--wavelet_s', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--threshold',type=float,default=1e-4)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='merge')
parser.add_argument('--normalize',type=bool,default=False)
# parser.add_argument('--laplacian_nomalize',type=bool,default=True)
parser.add_argument('--sparse_ness',type=bool,default=True)
parser.add_argument('--weight_normalize',type=bool,default=False)
parser.add_argument('--adj_scale',type=bool,default=True)
parser.add_argument('--reduction_ratio',type=int,default=8)
parser.add_argument('--dataset', type=str,default='dataset_instant_adj_d7')


args = parser.parse_args()
print(f'Training configs: {args}')
weight_normalize=args.weight_normalize
adj_scale=args.adj_scale
lr,batch_size,epoch=args.lr,args.batch_size,args.epoch
reduction_ratio=args.reduction_ratio
threshold,sparse_ness,s,n,n_his,n_pred=args.threshold,args.sparse_ness,args.wavelet_s,args.n_route,args.n_his,args.n_pred
data_name=args.dataset

#保存模型的路径
model_name='LSTSF-GWN'
out = 'out/%s'%(model_name)
#out = 'out/%s_%s'%(model_name,'perturbation')
path1 = '%s_%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r'%(model_name,data_name,lr,batch_size,n_his,n_pred,epoch)
save_path = os.path.join(out,path1)
if not os.path.exists(save_path):
    os.makedirs(save_path)

#模块数及模块层
blocks=[[1,32,64],[64,32,128]]

#不同数据集加载不同的邻接矩阵   
if data_name=='dataset_instant_adj_d7':
    #加载距离邻接矩阵
    try:
        adj_d= pd.read_csv(pjoin('./dataset/dataset_instant_adj_d7',f'PeMSD7_W_d_{n}.csv'), header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')
    if args.graph=='default':
        adj_d=weight_matrix(adj_d,adj_scale=True)
    else:
        adj_d=weight_matrix(pjoin('./dataset/dataset_instant_adj_d7',args.graph),adj_scale=True)
    
    LW1=wavelet_basis(adj_d,sparse_ness,threshold,weight_normalize,s)
    tf.add_to_collection(name='d_wavelet_basis1', value=tf.cast(tf.constant(LW1[0]), tf.float32))
    tf.add_to_collection(name='d_wavelet_basis2', value=tf.cast(tf.constant(LW1[1]), tf.float32))
    
    #加载相似度邻接矩阵
    try:
        adj_p= pd.read_csv(pjoin('./dataset/dataset_instant_adj_d7',f'PeMSD7_W_c_{n}.csv'), header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')
    if args.graph=='default':
        adj_p=weight_matrix(adj_p,adj_scale=False)
    else:
        adj_p=weight_matrix(pjoin('./dataset/dataset_instant_adj_d7',args.graph),adj_scale=False)
    LW2=wavelet_basis(adj_p,sparse_ness,threshold,weight_normalize,s=0.8)
    tf.add_to_collection(name='p_wavelet_basis1', value=tf.cast(tf.constant(LW2[0]), tf.float32))
    tf.add_to_collection(name='p_wavelet_basis2', value=tf.cast(tf.constant(LW2[1]), tf.float32))
    
    data_file = f'PeMSD7_V_{n}.csv'
    n_train, n_val, n_test = 34, 5, 5
    
    PeMS = data_gen(pjoin('./dataset/dataset_instant_adj_d7', data_file), (n_train, n_val, n_test), n, n_his + n_pred)
    print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')

if data_name=='dataset_instant_adj_d4':
    #加载距离邻接矩阵
    try:
        adj_d= pd.read_csv(pjoin('./dataset/dataset_instant_adj_d4',f'PeMSD7_W_d_{n}.csv'), header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')
    if args.graph=='default':
        adj_d=weight_matrix(adj_d,adj_scale=True)
    else:
        adj_d=weight_matrix(pjoin('./dataset/dataset_instant_adj_d4',args.graph),adj_scale=True)
    
    LW1=wavelet_basis(adj_d,sparse_ness,threshold,weight_normalize,s)
    tf.add_to_collection(name='d_wavelet_basis1', value=tf.cast(tf.constant(LW1[0]), tf.float32))
    tf.add_to_collection(name='d_wavelet_basis2', value=tf.cast(tf.constant(LW1[1]), tf.float32))
    
    #加载相似度邻接矩阵
    try:
        adj_p= pd.read_csv(pjoin('./dataset/dataset_instant_adj_d4',f'PeMSD4_W_c_{n}.csv'), header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')
    if args.graph=='default':
        adj_p=weight_matrix(adj_p,adj_scale=False)
    else:
        adj_p=weight_matrix(pjoin('./dataset/dataset_instant_adj_d7',args.graph),adj_scale=False)
    LW2=wavelet_basis(adj_p,sparse_ness,threshold,weight_normalize,s=0.8)
    tf.add_to_collection(name='p_wavelet_basis1', value=tf.cast(tf.constant(LW2[0]), tf.float32))
    tf.add_to_collection(name='p_wavelet_basis2', value=tf.cast(tf.constant(LW2[1]), tf.float32))
    
    data_file = f'PeMSD4_V_{n}.csv'
    n_train, n_val, n_test = 34, 5, 5
    
    PeMS = data_gen(pjoin('./dataset/dataset_instant_adj_d4', data_file), (n_train, n_val, n_test), n, n_his + n_pred)
    print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')

if __name__ == '__main__':
    model_train(PeMS, blocks, args,save_path)
    model_test(PeMS, PeMS.get_len('test'),n,n_his, n_pred, args.inf_mode,threshold,sparse_ness,s,weight_normalize,adj_scale,save_path)