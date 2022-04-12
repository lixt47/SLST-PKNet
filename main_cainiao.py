from utils.data_io import getGenerator
from utils.args import Args, list_of_param_dicts
from models.models import MLCNN
from models.model_runner import ModelRunner
import torch
import gc, os
torch.backends.cudnn.enabled = False
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import warnings 
warnings.filterwarnings('ignore')


param_dict = dict(
    data = ['./cainiao_data/1.csv',\
        './cainiao_data/2.csv',\
            './cainiao_data/3.csv',\
                './cainiao_data/4.csv',\
                    './cainiao_data/5.csv'],
    mode = ['continuous'],
    collaborate_span = [2],
    collaborate_stride = [1],
    train_share = [(0.8, 0.1)],
    input_T = [56],
    n_CNN = [5],
    kernel_size = [3],
    hidCNN = [50],
    hidRNN = [100],
    dropout = [0.2],
    highway_window = [56],
    clip = [10.],
    epochs = [100],
    batch_size =[8],
    seed = [54321],
    gpu = [0],
    cuda = [True],
    optim = ['adam'],
    lr = [0.001],
    L1Loss = [False],
    normalize = [0],
    skip = [7]
)

if __name__ == '__main__':
    params = list_of_param_dicts(param_dict)
    for param in params:
        cur_args = Args(param)
        generator = getGenerator(cur_args.data)
        data_gen = generator(cur_args.data, cur_args.mode, train_share=cur_args.train_share, input_T=cur_args.input_T,
                             collaborate_span=cur_args.collaborate_span, collaborate_stride=cur_args.collaborate_stride,
                             cuda=cur_args.cuda)
        runner = ModelRunner(cur_args, data_gen, None)
        runner.model = MLCNN(cur_args, data_gen)
        runner.run()
        runner.getMetrics()
        del runner
        gc.collect()
