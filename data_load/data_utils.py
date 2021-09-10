from utils.math_utils import z_score
import numpy as np
import pandas as pd

class Dataset(object):
    def __init__(self, data, stats):
        self.__data=data
        self.mean=stats['mean']
        self.std=stats['std']
    def get_data(self,type):
        return self.__data[type]
    def get_stats(self):
        return {'mean':self.mean,'std':self.std}
    def get_len(self,type):
        return len(self.__data[type])
    #这一步用来做什么？
    def z_inverse(self,type):
        return self.__data[type]*self.std+self.mean

def seq_gen(len_seq,data_seq,offset,n_frame,n_route,day_slot,c_0=1):
    #268=288-21+1
    n_slot=day_slot-n_frame+1
    #(34*268,21,228,1)
    tmp_seq=np.zeros((len_seq*n_slot,n_frame,n_route,c_0))
    for i in range(len_seq):
        for j in range(n_slot):
            sta=(i+offset)*day_slot+j
            end=sta+n_frame
            tmp_seq[i*n_slot+j,:,:,:]=np.reshape(data_seq[sta:end, :],[n_frame, n_route, c_0])
    return tmp_seq
                       
def data_gen(file_path,data_config,n_route,n_frame=27,day_slot=288):
    n_train,n_val,n_test=data_config #（34，5，5）
    try:
        data_seq=pd.read_csv(file_path,header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')
        
    seq_train=seq_gen(n_train,data_seq,0,n_frame,n_route,day_slot)
    seq_val = seq_gen(n_val, data_seq, n_train, n_frame, n_route, day_slot)
    seq_test = seq_gen(n_test, data_seq, n_train + n_val, n_frame, n_route, day_slot)
    #数据集值的和以及标准偏差
    x_stats={'mean':np.mean(seq_train),'std':np.std(seq_train)}
    
    #seq_train[33*268+267,21,228,1],一张图里有228个点，每个点有一个特征值，
    #对228个特征值进行求和和标准偏差，然后每个特征值做z_sorce处理
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])
    
    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    
    dataset = Dataset(x_data, x_stats)
    return dataset

#inputs是什么？batch_size=50
def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    '''
    Data iterator in batch.
    :param inputs: np.ndarray, [len_seq, n_frame, n_route, C_0], standard sequence units.
    :param batch_size: int, the size of batch.
    :param dynamic_batch: bool, whether changes the batch size in the last batch if its length is less than the default.
    :param shuffle: bool, whether shuffle the batches.
    '''
    len_inputs = len(inputs)
   

   #把数据集的顺序打乱
   #>>> arr = np.arange(10)
   #>>> np.random.shuffle(arr)
   #>> arr
   #[1 7 5 2 9 4 3 6 0 8]
    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)

        yield inputs[slide]
  #yield就是 return 返回一个值，并且记住这个返回的位置，下次迭代就从这个位置后(下一行)开始。  