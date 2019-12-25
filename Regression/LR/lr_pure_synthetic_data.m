AA = csvread('regression_pure_data.csv');
AA_syn = csvread('regression_pure_synthetic_data.csv');


[rr_real,cc_real] = size(AA)
[rr_real_syn, cc_real_syn] = size(AA_syn) 
num_runs = 100;

train_rmses = zeros(num_runs,1);
test_rmses = zeros(num_runs,1);

sum_train_rmse = 0;
sum_test_rmse = 0;
rng(0)
for i = 1:1:num_runs
  ind_arr = randperm(rr_real);
  
  XX_train = AA(ind_arr(1:122),1:101);
  
  XX_train = [XX_train; AA_syn(rr_real+1:end,1:101)]; 
  
  YY_train = AA(ind_arr(1:122),102:102);
  
  YY_train = [YY_train; AA_syn(rr_real+1:end,102:102)]; 
  
  [rr_train, cc_train] = size(XX_train);
  
  XX_test = AA(ind_arr(123:end),1:101);
  YY_test = AA(ind_arr(123:end),102:102);
  
  [rr_test, cc_test] = size(XX_test);
  FitForward = LinearModel.stepwise(XX_train,YY_train)
  Pred = predict (FitForward, XX_train);
  
 %beta = (XX_train' * XX_train)\(XX_train'*YY_train);
  
  train_rmse = sqrt(sum((Pred-YY_train).^2)/rr_train)
  
  
  Pred = predict (FitForward, XX_test);
  
  test_rmse = sqrt(sum((Pred-YY_test).^2)/rr_test)
  
  train_rmses(i,1) = train_rmse;
  test_rmses(i,1) = test_rmse;
end


fprintf('train rmse avg : %f +/- stddev: %f\n', mean(train_rmses), std  (train_rmses));

fprintf('test rmse avg : %f +/- stddev: %f\n', mean(test_rmses), std (test_rmses));