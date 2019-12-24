# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 11:34:02 2018

@author: P. Balamurugan
"""


#The code is suitable for multi-processing
#the following code is for xgradboost Classifier with k-fold cv for 100 different seed values and reports the average rmse along with standard deviation
#This code works on only pure data (without added synthetic data).  


import numpy as np
from sklearn.preprocessing import normalize #normalize is not used in the code, but can be tried
from sklearn.utils import shuffle 
import math 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import f1_score

import concurrent.futures
import time

global num_features
global crossvalidate_k 

global test_sample_index_list 

global seeds
global xGBparam_values
global splitpoints

global seed_subsample_pair_arr

global seed_start
global seed_step
global final_seed
global best_est 


test_sample_index_list = []
train_pure_synthetic_index_list = []
splitpoints = []

crossvalidate_k = 5 #number of folds for cross-validation

mydata = np.genfromtxt('classification_data.csv', delimiter=',')
num_features = len(mydata[0])-1

features = mydata[:,0:num_features]
output = mydata[:,num_features:]

norm_features = features




#print(norm_features)
#print(mydata)

num_samples = len(mydata)
#print(num_samples)


seed_start = 0.0
seed_end = 10000.0 #This is based on number of seeds 
seed_step = 100.0



seeds = np.arange(seed_start,seed_end,seed_step) 
#print(seeds)

final_seed = (seeds[len(seeds)-1])

test_rmses = np.zeros(len(seeds)) 
train_rmses = np.zeros(len(seeds))
accuracies = np.zeros(len(seeds))

best_xGBparam_values = 1.0*np.arange(0,len(seeds),1)

xGBparam_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #np.arange(100,200,100) 

seed_subsample_pair = []

for seed in seeds:
    for xGBparam_val in xGBparam_values:
        seed_subsample_pair.append([seed,xGBparam_val])

#print(seed_subsample_pair)
seed_subsample_pair_arr = np.array(seed_subsample_pair)
#print(seed_subsample_pair_arr[:,1])



estind_values = np.arange(len(xGBparam_values))

seed_indices = np.arange(len(seeds))

#print(num_estimators_values)
#print(estind_values)

#print(seeds)
#print(seed_indices)



def kfoldcv(seed,subsample):
    start = time.time()
    
    np.random.seed(int(seed))
    
    sample_index = np.arange(num_samples)
    #print(sampleindex)

    shuffled_indices = shuffle(sample_index)
    #print(shuffled_indices)
    
    test_proportion = 0.2  #set the proportion of test samples 
    num_test = int(test_proportion * num_samples) 
    #print(num_test)

    test_sample_index = shuffled_indices[:num_test]
    
    test_sample_index_list.append(test_sample_index)
    #print(test_sample_index)
    #print(len(test_sample_index))

    #split the remaining part into ten folds 
    train_validate_index = shuffled_indices[num_test:]
    #num_train_validate_samples = len(train_validate_index)
    #print(num_train_validate_samples)
    
    
    #print ('starting kfoldcv')
    subsample_arg = subsample

    fold_length = int(math.ceil((1.0*len(train_validate_index))/crossvalidate_k))
    splitpoints = np.arange(0,len(train_validate_index),fold_length)
    
    rmses = np.zeros(crossvalidate_k) 
    accuracies = np.zeros(crossvalidate_k)
    for i in np.arange(len(splitpoints)):
        #print(i)
        if i<len(splitpoints)-1:
            validate_index = train_validate_index[splitpoints[i]:splitpoints[i+1]]
        else:
            validate_index = train_validate_index[splitpoints[i]:]
        train_index = [x for x in train_validate_index if x not in validate_index]
        #print(validate_index)
        #print(len(validate_index))
        #print(train_index)
        #print(len(train_index))
        #print('**************************')


        train_feat = norm_features[train_index]
        train_feat = [np.reshape(x, (num_features, )) for x in train_feat]

        train_out = output[train_index]
        train_out = np.reshape(train_out, (len(train_out),))
        #print(train_data)

        #print('train')
        #print(i,np.shape(train_feat), np.shape(train_out))

        validate_feat = norm_features[validate_index]
        validate_feat = [np.reshape(x, (num_features, )) for x in validate_feat]

        validate_out = output[validate_index]
        validate_out = np.reshape(validate_out, (len(validate_out),))
        #print(train_data)

        #print('validate')
        #print(i,np.shape(validate_feat), np.shape(validate_out))

        #print(len(validate_samples))


        clf = GradientBoostingClassifier(subsample=subsample_arg, n_estimators=60)
     

        clf.fit(train_feat, train_out)
        

        #pred = clf.predict(train_feat)
        #tmp = ((x,y) for x,y in zip(pred, train_out))
        #print(list(tmp))


        pred = clf.predict(validate_feat)
        #tmp = ((x,y) for x,y in zip(pred, validate_out))
        #print(list(tmp))
        accuracy = metrics.accuracy_score(validate_out, pred)


        mse = sum((x-y)*(x-y) for x,y in zip(pred,validate_out))/len(validate_feat)
        rmse = np.sqrt(mse)

        #print(i,rmse)
        #print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        rmses[i] = rmse
        accuracies[i] = accuracy

        #if i==len(splitpoints)-1:
            #print(train_feat)
            #print(train_out)
            #print(validate_out)
        #print(rmses)
    avg_rmse = np.average(rmses)
    avg_accuracy = np.average(accuracies)
        
    return seed,subsample_arg,time.time() - start,avg_accuracy


def compute_testrmse(seed,best_param):
    start = time.time()
    
    np.random.seed(int(seed))
    
    sample_index = np.arange(num_samples)
    #print(sampleindex)

    shuffled_indices = shuffle(sample_index)
    #print(shuffled_indices)
    
    test_proportion = 0.2  #set the proportion of test samples 
    num_test = int(test_proportion * num_samples) 
    #print(num_test)

    test_sample_index = shuffled_indices[:num_test]
    
    test_sample_index_list.append(test_sample_index)
    #print(test_sample_index)
    #print(len(test_sample_index))

    #split the remaining part into ten folds 
    train_validate_index = shuffled_indices[num_test:]
    #num_train_validate_samples = len(train_validate_index)
    #print(num_train_validate_samples)
    
    #training set 
    final_train_feat = norm_features[train_validate_index]
    final_train_feat = [np.reshape(x, (num_features, )) for x in final_train_feat]

    final_train_out = output[train_validate_index]
    final_train_out = np.reshape(final_train_out, (len(final_train_out),))

    #test set 
    final_test_feat = norm_features[test_sample_index]
    final_test_feat = [np.reshape(x, (num_features, )) for x in final_test_feat]

    final_test_out = output[test_sample_index]
    final_test_out = np.reshape(final_test_out, (len(final_test_out),))


    final_best_param = best_param
    final_clf = GradientBoostingClassifier(subsample=final_best_param, n_estimators=60)
 
    final_clf.fit(final_train_feat, final_train_out)
    

    tr_pred = final_clf.predict(final_train_feat)
    final_tr_mse = sum((x-y)*(x-y) for x,y in zip(tr_pred,final_train_out))/len(final_train_feat)
    final_tr_rmse = np.sqrt(final_tr_mse)
    final_tr_accuracy = metrics.accuracy_score(final_train_out, tr_pred)

    #tmp = ((x,y) for x,y in zip(pred, train_out))
    #print(list(tmp))

    #pred = final_clf.predict()
    pred = final_clf.predict(final_test_feat)
    final_accuracy = metrics.accuracy_score(final_test_out, pred)
    #tmp = ((x,y) for x,y in zip(pred, final_test_out))
    #print(list(tmp))

    final_mse = sum((x-y)*(x-y) for x,y in zip(pred,final_test_out))/len(final_test_feat)
    final_rmse = np.sqrt(final_mse)

    return seed,final_best_param,time.time() - start, final_accuracy, final_tr_accuracy

    

def main():
    avg_rmse_ret = []
    xGBparam_ret = [] 
    
    for seed in seeds: 
        avg_rmse_ret.append([])
        xGBparam_ret.append([])    
    
    start = time.time()
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
         for seed,xGBparam,time_ret,avg_accuracy in executor.map(kfoldcv, seed_subsample_pair_arr[:,0], seed_subsample_pair_arr[:,1]):
             seed_index = int((seed-seed_start)/seed_step)
             avg_rmse_ret[seed_index].append(avg_accuracy)
             xGBparam_ret[seed_index].append(xGBparam)
             print('seed:%f subsample: %f time: %f avg_rmse: %f' %(seed, xGBparam,time_ret,avg_accuracy), flush=True)
    print('k fold cv completed ! Time taken: %f seconds' %(time.time()-start), flush=True )
    #print(xGBparam_ret)
    
    for seed in seeds: 
        seed_index = int((seed-seed_start)/seed_step)
        tmp = avg_rmse_ret[seed_index]
        #print(tmp)
        argminind=np.argmin(tmp)
        #print(argminind)
        
        estlist = xGBparam_ret[seed_index]
        #print(estlist)
        #print(estlist[argminind])
        best_xGBparam_values[seed_index]= (estlist[np.argmax(tmp)])
        
        print('seed:%d argmin subsample: %f' %(seed,best_xGBparam_values[seed_index]), flush=True)

    #print(best_est)

    start = time.time()        
    with concurrent.futures.ProcessPoolExecutor() as executor:
         for seed,best_xGBparam_val,time_ret,test_rmse,train_rmse in executor.map(compute_testrmse, seeds, best_xGBparam_values):
             print('seed:%f best subsample val:%f time: %f test_rmse: %f train rmse: %f' %(seed,best_xGBparam_val,time_ret,test_rmse, train_rmse), flush=True)
             seed_index = int((seed-seed_start)/seed_step)
             test_rmses[seed_index]=test_rmse
             train_rmses[seed_index]=train_rmse
    print('Test rmse computed ! Time taken: %f seconds' %(time.time()-start), flush=True )

    print('Test rmse: mean+/-std.dev of %d runs: %f +/- %f' %(len(seeds),np.average(test_rmses), np.std(test_rmses)) , flush=True)
    print('Train rmse: mean+/-std.dev of %d runs: %f +/- %f' %(len(seeds),np.average(train_rmses), np.std(train_rmses)) , flush=True)
    

if __name__ == '__main__':
    start = time.time()
    main()
    total_time = time.time() - start

    print('total time after completion: %f seconds' %(total_time), flush=True)



