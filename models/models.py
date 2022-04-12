"""
    Define all models.
"""

from pyexpat.errors import XML_ERROR_CANT_CHANGE_FEATURE_ONCE_PARSING
from utils._libs_ import torch, nn, F, np
from models.coint import Johansen
import statsmodels.tsa.arima_model as arima
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.tsatools import lagmat
import statsmodels.api as sm

# -------------------------------------------------------------------------------------------------------------------------------------------------
class MLCNN(nn.Module):
    def __init__(self, args, data):
        super(MLCNN, self).__init__()
        self.use_cuda = args.cuda
        self.input_T = args.input_T
        self.idim = data.column_num
        self.kernel_size = args.kernel_size
        self.hidC = args.hidCNN
        self.hidR = args.hidRNN
        self.hw = args.highway_window
        self.collaborate_span = args.collaborate_span
        self.cnn_split_num = int(args.n_CNN / (self.collaborate_span * 2 + 1))
        self.n_CNN = self.cnn_split_num * (self.collaborate_span * 2 + 1)
        self.skip = args.skip
        self.pt0 = (self.input_T - self.kernel_size) // self.skip
        self.pt1 = self.input_T // self.skip
        self.dropout = nn.Dropout(p = args.dropout)
        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        for i in range(self.n_CNN):
            if i == 0:
                tmpconv = nn.Conv2d(1, self.hidC, kernel_size=(self.kernel_size, self.idim), padding=(self.kernel_size//2, 0))
            else:
                tmpconv = nn.Conv2d(1, self.hidC, kernel_size=(self.kernel_size, self.hidC), padding=(self.kernel_size//2, 0))
            self.convs.append(tmpconv)
            self.bns.append(nn.BatchNorm2d(self.hidC))
        self.xy_convs = nn.ModuleList([])
        self.xy_bns = nn.ModuleList([])
        for i in range(self.n_CNN):
            if i == 0:
                tmpconv = nn.Conv2d(1, self.hidC, kernel_size=(self.kernel_size, 3+1), padding=(self.kernel_size//2, 0))
            else:
                tmpconv = nn.Conv2d(1, self.hidC, kernel_size=(self.kernel_size, self.hidC), padding=(self.kernel_size//2, 0))
            self.xy_convs.append(tmpconv)
            self.xy_bns.append(nn.BatchNorm2d(self.hidC))
        self.xyy_convs = nn.ModuleList([])
        self.xyy_bns = nn.ModuleList([])
        for i in range(self.n_CNN):
            if i == 0:
                tmpconv = nn.Conv2d(1, self.hidC, kernel_size=(self.kernel_size, self.idim+3), padding=(self.kernel_size//2, 0))
            else:
                tmpconv = nn.Conv2d(1, self.hidC, kernel_size=(self.kernel_size, self.hidC), padding=(self.kernel_size//2, 0))
            self.xyy_convs.append(tmpconv)
            self.xyy_bns.append(nn.BatchNorm2d(self.hidC))
        self.shared_lstm = nn.LSTM(self.hidC, self.hidR)
        self.target_lstm = nn.LSTM(self.hidC, self.hidR)
        self.shared_attn = nn.Sequential(
            nn.Linear(self.hidR, self.hidR),
            nn.Tanh(),
            nn.Linear(self.hidR, 1)
        )
        self.target_attn = nn.Sequential(
            nn.Linear(self.hidR, self.hidR),
            nn.Tanh(),
            nn.Linear(self.hidR, 1)
        )

        self.skip_shared_lstm = nn.LSTM(self.hidC, self.hidR)
        self.skip_target_lstm = nn.LSTM(self.hidC, self.hidR)

        self.xy_shared_lstm = nn.LSTM(self.hidC, self.hidR)
        self.xy_target_lstm = nn.LSTM(self.hidC, self.hidR)
        self.xy_shared_attn = nn.Sequential(
            nn.Linear(self.hidR, self.hidR),
            nn.Tanh(),
            nn.Linear(self.hidR, 1)
        )
        self.xy_target_attn = nn.Sequential(
            nn.Linear(self.hidR, self.hidR),
            nn.Tanh(),
            nn.Linear(self.hidR, 1)
        )

        self.xyy_shared_lstm = nn.LSTM(self.hidC, self.hidR)
        self.xyy_target_lstm = nn.LSTM(self.hidC, self.hidR)
        self.xyy_shared_attn = nn.Sequential(
            nn.Linear(self.hidR, self.hidR),
            nn.Tanh(),
            nn.Linear(self.hidR, 1)
        )
        self.xyy_target_attn = nn.Sequential(
            nn.Linear(self.hidR, self.hidR),
            nn.Tanh(),
            nn.Linear(self.hidR, 1)
        )


        self.linears = nn.ModuleList([])
        self.mul_linears = nn.ModuleList([])
        self.lstms = nn.ModuleList([])
        self.highways = nn.ModuleList([])
        for i in range(self.collaborate_span * 2 + 1):
            if self.skip > 0:
                self.linears.append(nn.Linear(self.hidR, 1))
                self.lstms.append(nn.LSTM(self.hidR * 3 + self.skip * self.hidR, self.hidR))
            else:
                self.linears.append(nn.Linear(self.hidR, 1))
                self.lstms.append(nn.LSTM(self.hidR * 3, self.hidR))
            if (self.hw > 0):
                self.highways.append(nn.Linear(self.hw, 1))
            self.mul_linears.append(nn.Linear(self.hidR, self.idim))
        
    
    def coint(self,x):
        x_idim=[]
        cox=[]
        for i in range(x.shape[0]): 
            cox.append([])
            x_idim.append([])
            j=12
            k=0
            while j<x.shape[2]:
                temp=Johansen(x[i][:,j-12:j])
                if temp.johansen()!=None:
                    x_idim[i].append([i for i in range(j-12,j)])
                    nx=x[i,:,x_idim[i][k]].cpu().numpy()
                    x_diff=np.diff(nx, axis=0)
                    x_diff_lags = lagmat(x_diff, 1, trim='both')
                    cox[i].append(x_diff_lags)
                    j=j+12
                    if k==4:
                        break
                    k=k+1
                else:
                    j=j+1
        return cox,x_idim


    # ---------------------------------------------------------------------------------------------------------------------------------------------
    def forward(self, x, f):
        cox,x_idim=self.coint(x)
        cohh=[]
        result=[]
        tst=[]
        temp=0
        for i in range(x.shape[0]):
            cohh.append([])
            result.append([])
            tst.append([])
            coxx=np.array(cox[i])
            for j in range(coxx.shape[0]):
                cohh[i].append([])
                temp=temp+1
                www=coxx[j]
                coARIMA=VAR(www)
                model_fit=coARIMA.fit()
                prediction = model_fit.forecast(www,steps=1)
                pre=torch.from_numpy(prediction).to(x.device)
                pre1=pre.float()
                tst[i].append(pre1)

        batch_size = x.size(0)
        f = torch.tensor(f,dtype=torch.float).to(x.device)
        
        # YY
        regressors = []
        currentR = torch.unsqueeze(x, 1)
        for i in range(self.n_CNN):
            currentR = self.convs[i](currentR)
            currentR = self.bns[i](currentR)
            currentR = F.leaky_relu(currentR, negative_slope=0.01)
            currentR = torch.squeeze(currentR, 3)
            if (i + 1) % self.cnn_split_num == 0:
                regressors.append(currentR)
                currentR = self.dropout(currentR)
            if i < self.n_CNN - 1:
                currentR = currentR.permute(0,2,1).contiguous()
                currentR = torch.unsqueeze(currentR, 1)

        shared_lstm_results = []
        target_R = None
        target_h = None
        target_c = None
        self.shared_lstm.flatten_parameters()
        for i in range(self.collaborate_span * 2 + 1):
            cur_R = regressors[i].permute(2,0,1).contiguous()
            cur_hidden, (cur_result, cur_state) = self.shared_lstm(cur_R)
            if i == self.collaborate_span:
                target_R = cur_R
                target_h = cur_result
                target_c = cur_state
            
            beta = F.softmax(self.shared_attn(cur_hidden.view(-1, self.hidR)).view(-1, self.input_T), dim=1)
            context = torch.bmm(beta.unsqueeze(1), cur_hidden.permute(1,0,2).contiguous())[:, 0, :]
            cur_result = self.dropout(context)
            shared_lstm_results.append(cur_result)
            

        self.target_lstm.flatten_parameters()
        target_hidden, (target_result, _) = self.target_lstm(target_R, (target_h, target_c))
        beta = F.softmax(self.target_attn(target_hidden.view(-1, self.hidR)).view(-1, self.input_T), dim=1)
        context = torch.bmm(beta.unsqueeze(1), target_hidden.permute(1,0,2).contiguous())[:, 0, :]
        target_result = self.dropout(context)

        #skip
        skip_shared_lstm_results = []
        skip_target_R = None
        skip_target_h = None
        skip_target_c = None
        self.skip_shared_lstm.flatten_parameters()
        for i in range(self.collaborate_span * 2 + 1):
            s = regressors[i][:, :, int(-self.pt0 * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt0, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt0, batch_size * self.skip, self.hidC)
            skip_cur_R = s
            _, (skip_cur_result, skip_cur_state) = self.skip_shared_lstm(skip_cur_R)
            if i == self.collaborate_span:
                skip_target_R = skip_cur_R
                skip_target_h = skip_cur_result
                skip_target_c = skip_cur_state
            skip_cur_result = self.dropout(torch.squeeze(skip_cur_result, 0))
            skip_cur_result = skip_cur_result.view(batch_size, self.skip * self.hidR)
            skip_shared_lstm_results.append(skip_cur_result)

        self.skip_target_lstm.flatten_parameters()
        _, (skip_target_result, _) = self.skip_target_lstm(skip_target_R, (skip_target_h, skip_target_c))
        skip_target_result = skip_target_result.view(batch_size, self.skip * self.hidR)
        skip_target_result = self.dropout(skip_target_result)


        # XY
        x1 = torch.unsqueeze(x, 3)
        f1 = torch.cat((x1, f), 3).permute(0,2,1,3).contiguous()
        f1 = f1.view(batch_size*self.idim, x.size(1), 4)

        xy_regressors = []
        currentR = torch.unsqueeze(f1, 1)
        for i in range(self.n_CNN):
            currentR = self.xy_convs[i](currentR)
            currentR = self.xy_bns[i](currentR)
            currentR = F.leaky_relu(currentR, negative_slope=0.01)
            currentR = torch.squeeze(currentR, 3)
            if (i + 1) % self.cnn_split_num == 0:
                xy_regressors.append(currentR)
                currentR = self.dropout(currentR)
            if i < self.n_CNN - 1:
                currentR = currentR.permute(0,2,1).contiguous()
                currentR = torch.unsqueeze(currentR, 1)

        xy_shared_lstm_results = []
        target_R = None
        target_h = None
        target_c = None
        self.xy_shared_lstm.flatten_parameters()
        for i in range(self.collaborate_span * 2 + 1):
            cur_R = xy_regressors[i].permute(2,0,1).contiguous()
            cur_hidden, (cur_result, cur_state) = self.xy_shared_lstm(cur_R)
            if i == self.collaborate_span:
                target_R = cur_R
                target_h = cur_result
                target_c = cur_state
            beta = F.softmax(self.xy_shared_attn(cur_hidden.view(-1, self.hidR)).view(-1, self.input_T), dim=1)
            context = torch.bmm(beta.unsqueeze(1), cur_hidden.permute(1,0,2).contiguous())[:, 0, :]
            cur_result = self.dropout(context)
            xy_shared_lstm_results.append(cur_result)

        self.xy_target_lstm.flatten_parameters()
        xy_target_hidden, (xy_target_result, _) = self.xy_target_lstm(target_R, (target_h, target_c))
        beta = F.softmax(self.xy_target_attn(xy_target_hidden.view(-1, self.hidR)).view(-1, self.input_T), dim=1)
        context = torch.bmm(beta.unsqueeze(1), xy_target_hidden.permute(1,0,2).contiguous())[:, 0, :]
        xy_target_result = self.dropout(context)


        # XYY
        x1 = torch.unsqueeze(x, 2).expand(batch_size, self.input_T, self.idim, self.idim)
        f2 = torch.cat((x1, f), 3).permute(0,2,1,3).contiguous()
        f2 = f2.view(batch_size*self.idim, x.size(1), self.idim+3)

        xyy_regressors = []
        currentR = torch.unsqueeze(f2, 1)
        for i in range(self.n_CNN):
            currentR = self.xyy_convs[i](currentR)
            currentR = self.xyy_bns[i](currentR)
            currentR = F.leaky_relu(currentR, negative_slope=0.01)
            currentR = torch.squeeze(currentR, 3)
            if (i + 1) % self.cnn_split_num == 0:
                xyy_regressors.append(currentR)
                currentR = self.dropout(currentR)
            if i < self.n_CNN - 1:
                currentR = currentR.permute(0,2,1).contiguous()
                currentR = torch.unsqueeze(currentR, 1)

        xyy_shared_lstm_results = []
        target_R = None
        target_h = None
        target_c = None
        self.xyy_shared_lstm.flatten_parameters()
        for i in range(self.collaborate_span * 2 + 1):
            cur_R = xyy_regressors[i].permute(2,0,1).contiguous()
            _, (cur_result, cur_state) = self.xyy_shared_lstm(cur_R)
            if i == self.collaborate_span:
                target_R = cur_R
                target_h = cur_result
                target_c = cur_state
            beta = F.softmax(self.xyy_shared_attn(cur_hidden.view(-1, self.hidR)).view(-1, self.input_T), dim=1)
            context = torch.bmm(beta.unsqueeze(1), cur_hidden.permute(1,0,2).contiguous())[:, 0, :]
            cur_result = self.dropout(context)
            xyy_shared_lstm_results.append(cur_result)

        self.xyy_target_lstm.flatten_parameters()
        xyy_target_hidden, (xyy_target_result, _) = self.xyy_target_lstm(target_R, (target_h, target_c))
        beta = F.softmax(self.xyy_target_attn(xyy_target_hidden.view(-1, self.hidR)).view(-1, self.input_T), dim=1)
        context = torch.bmm(beta.unsqueeze(1), xyy_target_hidden.permute(1,0,2).contiguous())[:, 0, :]
        xyy_target_result = self.dropout(context)

        res = None
        for i in range(self.collaborate_span * 2 + 1):
            if i == self.collaborate_span:
                yy_target_result_ = torch.unsqueeze(target_result, 1).expand(batch_size, self.idim, self.hidR)
                xy_target_result_ = xy_target_result.view(batch_size, self.idim, self.hidR)
                xyy_target_result_ = xyy_target_result.view(batch_size, self.idim, self.hidR)
                skip_target_result_ = torch.unsqueeze(skip_target_result, 1).expand(batch_size, self.idim, self.skip * self.hidR)
                inp = torch.cat((yy_target_result_, xy_target_result_, xyy_target_result_, skip_target_result_), 2).permute(1,0,2).contiguous()
                inp1, _ = self.lstms[i](inp)
                cur_res = self.linears[i](inp1.permute(1,0,2).contiguous())
                cur_res = torch.squeeze(cur_res, 2)

                mul_cur_res = self.mul_linears[i](target_result)
                cur_res = cur_res + mul_cur_res

                for p in range(x.shape[0]):
                    for j in range(np.array(tst[p]).shape[0]):
                        temm=tst[p][j].squeeze()
                        cur_res[p,x_idim[p][j]]=cur_res[p,x_idim[p][j]]+temm
            else:
                yy_shared_lstm_results_ = torch.unsqueeze(shared_lstm_results[i], 1).expand(batch_size, self.idim, self.hidR)
                xy_shared_lstm_results_ = xy_shared_lstm_results[i].view(batch_size, self.idim, self.hidR)
                xyy_shared_lstm_results_ = xyy_shared_lstm_results[i].view(batch_size, self.idim, self.hidR)
                skip_shared_lstm_results_ = torch.unsqueeze(skip_shared_lstm_results[i], 1).expand(batch_size, self.idim, self.skip * self.hidR)
                inp = torch.cat((yy_shared_lstm_results_, xy_shared_lstm_results_, xyy_shared_lstm_results_, skip_shared_lstm_results_), 2).permute(1,0,2).contiguous()
                inp1, _ = self.lstms[i](inp)
                cur_res = self.linears[i](inp1.permute(1,0,2).contiguous())
                cur_res = torch.squeeze(cur_res, 2)

                mul_cur_res = self.mul_linears[i](shared_lstm_results[i])
                cur_res = cur_res + mul_cur_res

            cur_res = torch.unsqueeze(cur_res, 1)
            if res is not None:
                res = torch.cat((res, cur_res), 1)
            else:
                res = cur_res
        

        if (self.hw > 0):
            highway = None
            for i in range(self.collaborate_span * 2 + 1):
                z = x[:, -self.hw:, :]
                z = z.permute(0,2,1).contiguous().view(-1, self.hw)
                z = self.highways[i](z)
                z = z.view(-1, self.idim)
                z = torch.unsqueeze(z, 1)
                if highway is not None:
                    highway = torch.cat((highway, z), 1)
                else:
                    highway = z
            res = res + highway

        return res
