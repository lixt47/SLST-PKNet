"""
    Implements all io operations of data.
"""

from utils._libs_ import np, pd, torch, Variable
import json
from utils.trans_data import feature_eng 

# -------------------------------------------------------------------------------------------------------------------------------------------------

"""
Get the data generator
"""
def getGenerator(data_name):
    return GeneralGenerator

# -------------------------------------------------------------------------------------------------------------------------------------------------
"""
DataGenerator class produces data samples for all models
"""
class DataGenerator():
    def __init__(self, data_dict, mode, train_share=(.6, .2), input_T=56, collaborate_span=0,
                 collaborate_stride=1, limit=np.inf, cuda=False):
        if mode == "continuous" and collaborate_span <= 0:
            raise Exception("collaborate_span must > 0!")

        self.data_dict=data_dict
        if limit < np.inf: self.X = self.X[:limit]
        self.train_share = train_share
        self.input_T = input_T
        self.collaborate_span = collaborate_span
        self.collaborate_stride = collaborate_stride
        self.column_num = np.array(self.data_dict[0]['X']).shape[1]
        self.F_column_num = np.array(self.data_dict[0]['F']).shape[2]
        self.mode = mode
        self.cuda = cuda

        self.split_data()

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Split the training, validation and testing data
    """
    def split_data(self):
        test_range = int(len(self.data_dict) * (1-self.train_share[0]-self.train_share[1])) - 1
        valid_range = test_range + int(len(self.data_dict) * self.train_share[1]) - 1
        train_range = len(self.data_dict)
        train_dict, valid_dict, test_dict = [], [], []
        for d in self.data_dict:
            if d['watch'] < test_range:
                test_dict.append(d)
            elif d['watch'] < valid_range:
                valid_dict.append(d)
            else:
                train_dict.append(d)
    
        self.train_set = self.batchify(train_dict, valid_dict)
        self.valid_set = self.batchify(valid_dict, test_dict)
        self.test_set = self.batchify(test_dict, test_dict, 'test')

    # ---------------------------------------------------------------------------------------------------------------------------------------------

    def batchify(self, setDict, testdict, mode='train'):
        idx_num = np.array(setDict).shape[0]
        if self.mode == "immediate":
            X = torch.zeros((idx_num, self.input_T, self.column_num))
            F = torch.zeros((idx_num, self.input_T, self.column_num, self.F_column_num))
            Y = torch.zeros((idx_num, self.column_num))
            for i in range(idx_num):
                X[i, :, :] = torch.from_numpy(self.setDict[i]['X'])
                F[i, :, :, :] = torch.from_numpy(self.setDict[i]['F'])
                Y[i, :] = torch.from_numpy(self.setDict[i]['y'])
        elif self.mode == "continuous":
            X = torch.zeros((idx_num, self.input_T, self.column_num))
            F = torch.zeros((idx_num, self.input_T, self.column_num, self.F_column_num))
            Y = torch.zeros((idx_num, self.collaborate_span * 2 + 1, self.column_num))
            for i in range(idx_num):
                X[i, :, :] = torch.from_numpy(np.array(setDict[i]['X']))
                F[i, :, :, :] = torch.from_numpy(np.array(setDict[i]['F']))
                Y[i, 0:self.collaborate_span, :] = torch.from_numpy(np.array(setDict[i]['X'][-self.collaborate_span-1:-1]))
                Y[i, self.collaborate_span, :] = torch.from_numpy(np.array(setDict[i]['y']))
                if mode == 'test':
                    Y[i, self.collaborate_span+1:self.collaborate_span * 2 + 1, :] = torch.from_numpy(np.array(setDict[i]['y']))
                else:
                    if i!=np.array(setDict).shape[0]-1:
                        Y[i, self.collaborate_span+1:self.collaborate_span * 2 + 1, :] = torch.from_numpy(np.array(setDict[i+1]['X'][0:self.collaborate_span]))
                    else:
                        Y[i, self.collaborate_span+1:self.collaborate_span * 2 + 1, :] = torch.from_numpy(np.array(testdict[0]['X'][0:self.collaborate_span]))
        else:
            raise Exception('invalid mode')
        return [X, F, Y]


    # ---------------------------------------------------------------------------------------------------------------------------------------------
    def get_batches(self, X, F, Y, batch_size, shuffle=True):
        length = len(X)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            batch_X = X[excerpt]
            batch_F = F[excerpt]
            batch_Y = Y[excerpt]
            if (self.cuda):
                batch_X = batch_X.cuda()
                batch_F = batch_F.cuda()
                batch_Y = batch_Y.cuda()
            yield Variable(batch_X), Variable(batch_F),Variable(batch_Y)
            start_idx += batch_size

# -------------------------------------------------------------------------------------------------------------------------------------------------
class GeneralGenerator(DataGenerator):
     def __init__(self, data_path, mode, train_share=(.6, .2), input_T=56, collaborate_span=0,
                  collaborate_stride=1, limit=np.inf, cuda=False):
        data_dict = feature_eng(data_path, input_T)
        super(GeneralGenerator, self).__init__(data_dict, mode=mode,
                                               train_share=train_share,
                                               input_T=input_T,
                                               collaborate_span=collaborate_span,
                                               collaborate_stride=collaborate_stride,
                                               limit=limit,
                                               cuda=cuda)

