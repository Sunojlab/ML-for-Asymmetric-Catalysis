# -*- coding: utf-8 -*-
"""

@author: P. Balamurugan
"""

import numpy as np
from torch import nn
import torch
from torch.autograd import Variable

import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle 

import sys

global f


#print(sys.argv[1])

np.random.seed(int(sys.argv[1]) )

f = open("368_101_pure_syn_data_logs.txt", "a")


class CNN1D_test(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, n_kernels=5):
        super(CNN1D_test, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        
        self.hidden_2_size = 128
        self.hidden_3_size = 64
        self.hidden_4_size = 32
        self.dropout_prob = 0.5
        
        self.n_kernels = n_kernels
        
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), padding=(0, 0)) 
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 1), padding=(0, 0)) 
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 1), padding=(0, 0)) 
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(4, 1), padding=(0, 0)) 
        self.conv5 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 1), padding=(0, 0)) 
        

        self.dropout = nn.Dropout(self.dropout_prob)
        
        self.linearinputsize  = 0 
        for i in range(self.n_kernels):
            self.linearinputsize += (input_size-i)
        
        print(self.linearinputsize)
        
        self.fc2 = nn.Linear(self.linearinputsize, self.hidden_3_size)
        self.sig2 = nn.LogSigmoid()
        
        self.conv_seq = nn.Sequential(         # input shape (1, 28, 28)
            nn.Linear(self.hidden_3_size, self.output_size),
        )
        

    def forward(self, input_c1, input_c2, input_c3, input_c4, input_c5):
        c1 = self.conv1(input_c1)
        c2 = self.conv2(input_c2)
        c3 = self.conv3(input_c3)
        c4 = self.conv4(input_c4)
        c5 = self.conv5(input_c5)
        
        #print(c1.size())
        #print(c2.size())
        #print(c3.size())
        #print(c4.size())
        #print(c5.size())
        
        
        
        combined  = torch.cat([c1,  torch.cat([c2, torch.cat([c3, torch.cat([c4,c5],dim=2)], dim=2)],dim=2)], dim=2)
        #print(combined.size())
        
        #print(combined.size())
        combined = combined.view((1,(combined.size())[2]))
        #print(combined.size())
        
        combined = self.fc2(combined)
        combined = self.sig2(combined)
        
        output = self.conv_seq(combined)
        return (output)

def evaluate_loss(cnn_obj, test_data, test_labels, criterion):
    #for x,y in test_data:
    #    print(feed_forward_eval(biases,weights,x))
    num_test_samples = len(test_data)
    num_features = len(test_data[0])
    
    print(num_test_samples)
    running_test_loss = 0
    for j in range(num_test_samples): 
        T = torch.Tensor(test_data[j])
        T = T.view(1,1,num_features,1)

        test_inputs = Variable(T)
        y_pred = cnn_obj(test_inputs, test_inputs, test_inputs, test_inputs, test_inputs)
        
        actual_output = torch.Tensor(test_labels[j])
        actual_output = actual_output.view(1,1)

        
        test_loss = criterion(y_pred, actual_output)
        running_test_loss += test_loss.item()
        
    test_rmse = np.sqrt(running_test_loss/num_test_samples)
    print('current test loss: %f' %(test_rmse))
    return test_rmse     


mydata = np.genfromtxt('regression_pure_data.csv', delimiter=',')

num_samples = len(mydata)

num_features = len(mydata[0])-1

features = mydata[:,0:num_features]
output = mydata[:,num_features:]

norm_features = features 


mydata_withsynthetic = np.genfromtxt('regression_pure_synthetic_data.csv', delimiter=',')

pure_synthetic_features = mydata_withsynthetic[:,0:num_features]
pure_synthetic_output = mydata_withsynthetic[:,num_features:]


input_size = num_features
hidden_size = 700
output_size = 1
n_layers = 1
n_kernels = 5

sample_index = np.arange(num_samples)
#print(sampleindex)

shuffled_indices = shuffle(sample_index)
#print(shuffled_indices)

test_proportion = 0.2  #set the proportion of test samples 
num_test = int(test_proportion * num_samples) 
#print(num_test)

test_sample_index = shuffled_indices[:num_test]


#split the remaining part into ten folds 
train_validate_index = shuffled_indices[num_test:]


num_synthetic_samples = len(mydata_withsynthetic) - len(mydata) #note: the new file contains original data and synthetic data 
new_synthetic_indices = np.arange(num_samples, num_samples+num_synthetic_samples,1)
#print(new_synthetic_indices)

train_validate_puresynthetic_index = np.concatenate( (train_validate_index, new_synthetic_indices) , axis=None)

#print('run:%d num estimators: %d avg. rmse: %f' %(run,num_estimators_arg, avg_rmse))

#training set 
final_train_feat = pure_synthetic_features[train_validate_puresynthetic_index]
#final_train_feat = [np.reshape(x, (num_features, )) for x in final_train_feat]

final_train_out = pure_synthetic_output[train_validate_puresynthetic_index]
#final_train_out = np.reshape(final_train_out, (len(final_train_out),))

#test set 
final_test_feat = features[test_sample_index]

final_test_out = output[test_sample_index]

print(np.shape(final_test_feat))
print(np.shape(final_test_out))

np.set_printoptions(threshold=np.inf)

#final_train_feat = features[train_validate_index]
num_train_samples = len(final_train_feat)
#final_train_out = output[train_validate_index]
    

cnn = CNN1D_test(input_size, hidden_size, output_size, n_layers=n_layers, n_kernels=n_kernels)
print(cnn)

criterion = nn.MSELoss()

optimizer = optim.SGD(cnn.parameters(), lr=1e-7, momentum=0)

for epoch in range(10001):  # loop over the dataset multiple times

    if epoch==0 or (epoch+1) % 100 ==0:
        print('*******************************')
        print('epoch: %d' %(epoch+1))
    running_loss = 0.0
    ind_arr = np.arange(0,num_train_samples)
    np.random.shuffle(ind_arr)
    
    for ind in range(num_train_samples):
        
        i = ind_arr[ind]
        
        T = torch.Tensor(final_train_feat[i])
        T = T.view(1,1,num_features,1)


        inputs = Variable(T)
        y_pred = cnn(inputs, inputs, inputs, inputs, inputs)
        #print('y pred:')
        #print(y_pred)
        optimizer.zero_grad()

        
        # forward + backward + optimize
        #actual_output = Variable(torch.Tensor(output[i]))
        
        actual_output = torch.Tensor(final_train_out[i])
        actual_output = actual_output.view(1,1)
        
        loss = criterion(y_pred, actual_output)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        #print('current loss: %f running loss: %f' %(loss.item(),running_loss))
    if epoch==0 or (epoch+1) % 100 ==0:
        intermediate_train_rmse = np.sqrt(running_loss/num_samples)
        intermediate_test_rmse = evaluate_loss(cnn, final_test_feat, final_test_out, criterion)
        print("epoch:%d test_rmse: %f train_rmse: %f\n" %((epoch+1),intermediate_test_rmse,intermediate_train_rmse))
        f.write("epoch:%d test_rmse: %f train_rmse: %f\n" %((epoch+1),intermediate_test_rmse,intermediate_train_rmse))
        f.flush()

        #print('current running loss: %f' %(np.sqrt(running_loss/num_samples)))
        #evaluate_loss(cnn, final_test_feat, final_test_out, criterion)
        print('***************************')
