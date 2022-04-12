
import csv
from utils._libs_ import math, time, torch, nn, np
from utils.data_io import DataGenerator
from models.optimize import Optimize
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

class ModelRunner():
    def __init__(self, args, data_gen, model):
        self.args = args
        self.data_gen = data_gen
        self.model = model
        self.best_rmse = None
        self.best_rse = None
        self.best_mae = None
        self.best_corr = None
        self.running_times = []
        self.train_losses = []
        self.predictlast=[]

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    def huber(true, pred, delta):
        loss = np.where(np.abs(true-pred) < delta , 0.5*((true-pred)**2), delta*np.abs(true - pred) - 0.5*(delta**2))
        return np.sum(loss)
    """
    Train the model
    """
    def train(self):
        self.model.train()
        total_loss = 0
        n_samples = 0

        for X, F, Y in self.data_gen.get_batches(self.data_gen.train_set[0], self.data_gen.train_set[1], self.data_gen.train_set[2], self.args.batch_size, True):
            self.model.zero_grad()
            output = self.model(X, F)
            loss = self.criterion(output, Y)
            loss.backward()
            grad_norm = self.optim.step()
            total_loss += loss.item()
            if self.data_gen.mode == "immediate":
                n_samples += (output.size(0) * self.data_gen.column_num)
            else:
                n_samples += (output.size(0) * output.size(1) * self.data_gen.column_num)

        return total_loss / n_samples

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    def evaluate(self, mode='valid'):
        self.model.eval()
        total_loss = 0
        total_loss_l1 = 0
        n_samples = 0
        predict = None
        test = None

        if mode == 'train':
            tmp_X = self.data_gen.train_set[0]
            tmp_F = self.data_gen.train_set[1]
            tmp_Y = self.data_gen.train_set[2]
        elif mode == 'valid':
            tmp_X = self.data_gen.valid_set[0]
            tmp_F = self.data_gen.valid_set[1]
            tmp_Y = self.data_gen.valid_set[2]
        elif mode == 'test':
            tmp_X = self.data_gen.test_set[0]
            tmp_F = self.data_gen.test_set[1]
            tmp_Y = self.data_gen.test_set[2]
            
        else:
            raise Exception('invalid evaluation mode')

        mae_list, rmae_list, rse_list,corr_list = [], [], [], []
        for X, F, Y in self.data_gen.get_batches(tmp_X, tmp_F, tmp_Y, self.args.batch_size, False):
            output = self.model(X, F)
            L1_loss = self.evaluateL1(output, Y).item()
            L2_loss = self.evaluateL2(output, Y).item()
            for i in range(output.size(0)):
                predict_list = output[i, self.data_gen.collaborate_span].data.cpu().numpy()
                true_list = Y[i, self.data_gen.collaborate_span].data.cpu().numpy()
                mae = np.sum(np.absolute(predict_list-true_list))/len(true_list)
                rmae = mae / np.mean(true_list)
                rse = np.sqrt(mean_squared_error(true_list, predict_list)) / np.std(true_list, ddof=1)
                sigma_t = np.std(true_list, ddof=1)
                sigma_p = np.std(predict_list, ddof=1)
                corr = np.mean((true_list - np.mean(true_list)) * (predict_list - np.mean(predict_list))) / (sigma_t * sigma_p)
                mae_list.append(mae)
                rmae_list.append(rmae)
                rse_list.append(rse)
                corr_list.append(corr)
            if predict is None:
                predict = output
                test = Y
            else:
                predict = torch.cat((predict, output))
                test = torch.cat((test, Y))
            total_loss_l1 += L1_loss
            total_loss += L2_loss
            n_samples += (output.size(0) * self.data_gen.column_num)

        mse = total_loss / n_samples
        mae = np.mean(mae_list)
        rmae = np.mean(rmae_list)
        rse = np.mean(rse_list)
        correlation = np.mean(corr_list)
        return mse, rse, rmae, correlation

    def evaluate1(self):
        self.model.eval()
        predict = None
        X = self.data_gen.test_set[0]
        F = self.data_gen.test_set[1]
        X=torch.tensor(X, dtype=torch.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X=X.to(device)
        output = self.model(X, F)
        predict = output[:,self.data_gen.collaborate_span]
        Y = self.data_gen.test_set[2]

        predict_list = predict.data.cpu().numpy()
        true_list = Y.data.cpu().numpy()
        mae = np.sum(np.absolute(predict_list-true_list))/len(true_list)
        rmae = mae / np.mean(true_list)
        rse = np.sqrt(mean_squared_error(true_list, predict_list)) / np.std(true_list, ddof=1)
        sigma_t = np.std(true_list, ddof=1)
        sigma_p = np.std(predict_list, ddof=1)
        corr = np.mean((true_list - np.mean(true_list)) * (predict_list - np.mean(predict_list))) / (sigma_t * sigma_p)

        return mse, rse, rmae, corr

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    
    def run(self):
        use_cuda = self.args.gpu is not None
        if use_cuda:
            if type(self.args.gpu) == list:
                self.model = nn.DataParallel(self.model, device_ids=self.args.gpu)
            else:
                torch.cuda.set_device(self.args.gpu)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(self.args.seed)
        if use_cuda: self.model.cuda()

        self.nParams = sum([p.nelement() for p in self.model.parameters()])

        if self.args.L1Loss:
            self.criterion = nn.L1Loss(reduction='sum')
        else:
            self.criterion = nn.SmoothL1Loss(reduction='sum')
        self.evaluateL1 = nn.L1Loss(reduction='sum')
        self.evaluateL2 = nn.MSELoss(reduction='sum')
        if use_cuda:
            self.criterion = self.criterion.cuda()
            self.evaluateL1 = self.evaluateL1.cuda()
            self.evaluateL2 = self.evaluateL2.cuda()

        self.optim = Optimize(self.model.parameters(), self.args.optim, self.args.lr, self.args.clip)

        best_valid_mse = float("inf")
        best_valid_rse = float("inf")
        best_valid_mae = float("inf")
        best_valid_corr = -float("inf")
        best_test_mse = float("inf")
        best_test_rse = float("inf")
        best_test_mae = float("inf")
        best_test_corr = -float("inf")

        tmp_losses = []
        try:
            for epoch in range(1, self.args.epochs+1):
                epoch_start_time = time.time()
                train_loss = self.train()
                self.running_times.append(time.time() - epoch_start_time)
                tmp_losses.append(train_loss)
                with torch.no_grad():
                    val_mse, val_rse, val_mae, val_corr = self.evaluate()
                if val_mse < best_valid_mse:
                    best_valid_mse = val_mse
                if val_rse < best_valid_rse:
                    best_valid_rse = val_rse
                if val_mae < best_valid_mae:
                    best_valid_mae = val_mae
                if val_corr < best_valid_corr:
                    best_valid_corr = val_corr
                '''
                with torch.no_grad():
                    test_mse, test_rse, test_mae, test_corr = self.evaluate(mode='test')
                if test_mae < best_test_mae:
                    best_test_mse = test_mse
                    best_test_rse = test_rse
                    best_test_mae = test_mae
                    best_test_corr = test_corr
                '''
                self.optim.updateLearningRate(val_mse, epoch)
                    
        except KeyboardInterrupt:
            pass
        with torch.no_grad():
            test_mse, test_rse, test_mae, test_corr = self.evaluate(mode='test')
            best_test_mse = test_mse
            best_test_rse = test_rse
            best_test_mae = test_mae
            best_test_corr = test_corr
        

        self.best_rmse = np.sqrt(best_valid_mse)
        self.best_rse = best_valid_rse
        self.best_mae = best_valid_mae
        self.best_corr = best_valid_corr

        self.test_rmse = np.sqrt(best_test_mse)
        self.test_rse = best_test_rse
        self.test_mae = best_test_mae
        self.test_corr = best_test_corr

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Compute and output the metrics
    """
    def getMetrics(self):
        print('-' * 100)
        print()

        print('* number of parameters: %d' % self.nParams)
        for k in self.args.__dict__.keys():
            print(k, ': ', self.args.__dict__[k])

        running_times = np.array(self.running_times)
        train_losses = np.array(self.train_losses)
        #pd.DataFrame(running_times).to_csv('time_1.csv')
        #pd.DataFrame(train_losses).to_csv('loss_1.csv')

        print("time: sum {:8.7f} | mean {:8.7f}".format(np.sum(running_times), np.mean(running_times)))
        #print("training time: ", self.running_times)
        #print("loss trend: ", self.train_losses)
        print("test rmse: {:8.7f}".format(self.test_rmse))
        print("test rse: {:8.7f}".format(self.test_rse))
        print("test mae: {:8.7f}".format(self.test_mae))
        print("test corr: {:8.7f}".format(self.test_corr))
        print()

