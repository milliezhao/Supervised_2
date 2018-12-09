import numpy as np
from collections import Counter
import itertools
import matplotlib.pyplot as plt
import random
import copy
from sklearn import metrics

def reduce_dataset(data, number):
    '''A function to reduce the MNIST dataset by taking 100 random samples of each digit'''
    
    Y=data[:,0]
    labels_set=list(set(Y))
    
    #Get index for each label
    indices = { value : [ i for i, v in enumerate(Y) if v == value]
            for value in labels_set}
    
    #Get 100 uniform samples for each label
    reduced_indices={}
    for i in labels_set:
        sample_index=np.random.choice(indices[i], size=int(number/10), replace=False, p=None)
        reduced_indices[i]=sample_index
        
    reduced_data_index=np.array(list(reduced_indices.values())).ravel()
    np.random.shuffle(reduced_data_index)
    
    reduced_dataset=data[reduced_data_index,:]
    
    return reduced_dataset

def randomly_split_data(data, train_fra):
    train_no= int(len(data)*train_fra)
    index=np.arange(0,len(data),1)
    np.random.shuffle(index)
    train_index=index[:train_no]
    test_index=np.setdiff1d(index, train_index)
    
    return train_index, test_index

def Poly_Kernals(X,T,d):
    '''Produce the Gram Matrix K of the Polynomial Kernel
    X: shape N1xK
    T: shape N2xK
    d: degree of the polynomial
    K: shape N1xN2
    '''

    N_1,K=X.shape
    N_2,K=T.shape
    
    Copy_X = np.repeat(X[:, :, np.newaxis], N_2, axis=2)
    Copy_T = np.repeat(T[:, :, np.newaxis], N_1, axis=2)
    
    New_T=np.einsum("ijk->kji",Copy_T)
    
    K0=np.einsum("ijk,ijk->ijk",Copy_X,New_T)
    
    K=np.sum(K0,axis=1)**d

    return K

def Gaussian_Kernals(X,T,c):
    '''Produce the Gram Matrix K of the Guassian Kernel
    X: shape N1xK
    T: shape N2xK
    c: width of the gaussian kernel
    K: shape N1xN2
    '''
    
    N_1,K=X.shape
    N_2,K=T.shape
    
    Copy_X = np.repeat(X[:, :, np.newaxis], N_2, axis=2)
    Copy_T = np.repeat(T[:, :, np.newaxis], N_1, axis=2)
    
    New_T=np.einsum("ijk->kji",Copy_T)
    
    Diff=Copy_X-New_T
    Diff_squared=np.einsum("ijk,ijk->ijk",Diff,Diff)
    Diff_squared_sum=np.sum(Diff_squared,axis=1)
    D=np.divide(Diff_squared_sum,2.)
    K=np.exp(-c*D)

    return K

def kernalise_data(train_data, val_data, test_data, n_class, n_epoch, param, kernel='poly'):
    '''
    A function to produce data of the correct form ready to use in kernel perceptron by
    kernelising X and one-hot encoding Y, followed by extending the data according to the number
    of trainning epoch. If Kernel=None, then kernel is not applied.

    train_data: No_train x K
    test_data: No_test x K
    K_long: (No_train*n_epoch) x (No_train*n_epoch)
    Y_one_hot_long: (No_train*n_epoch) x n_class
    K_long_test: (No_train*n_epoch) x No_test
    Y_one_hot_test: No_test x n_class
    '''

    K_long=None
    Y_one_hot_long=None
    K_long_val=None
    Y_one_hot_val=None
    K_long_test=None
    Y_one_hot_test=None
    
    if train_data is not None:
        X=train_data[:,1:]
        Y=train_data[:,0]
        
        #One hot encoding Y
        Y_one_hot = np.zeros((len(Y), n_class))
        Y_one_hot[np.arange(len(Y)), list(map(int,Y))] = 1
        
        #Make duplicate for training with multiple epochs
        Y_one_hot_long=np.tile(Y_one_hot.T,n_epoch).T

        if kernel=='poly':
            K=Poly_Kernals(X,X,param)
            A=np.tile(K.T,n_epoch).T
            K_long=np.tile(A,n_epoch)
            
        elif kernel=='gauss':
            K=Gaussian_Kernals(X,X,param)       
            A=np.tile(K.T,n_epoch).T
            K_long=np.tile(A,n_epoch)
            
        else:
            K_long=np.tile(X.T,n_epoch).T
    
    if val_data is not None:
        val_X=val_data[:,1:]
        val_Y=val_data[:,0]
        Y_one_hot_val = np.zeros((len(val_Y), n_class))
        Y_one_hot_val[np.arange(len(val_Y)), list(map(int,val_Y))] = 1
        
        if kernel=='poly':
            val_K=Poly_Kernals(X,val_X,param)
            K_long_val=np.tile(val_K.T,n_epoch).T
            
        elif kernel=='gauss':
            val_K=Gaussian_Kernals(X,val_X,param)
            K_long_val=np.tile(val_K.T,n_epoch).T
            
        else:
            K_long_val=val_X
            
    if test_data is not None:
        test_X=test_data[:,1:]
        test_Y=test_data[:,0]
        Y_one_hot_test = np.zeros((len(test_Y), n_class))
        Y_one_hot_test[np.arange(len(test_Y)), list(map(int,test_Y))] = 1
        
        if kernel=='poly':
            test_K=Poly_Kernals(X,test_X,param)
            K_long_test=np.tile(test_K.T,n_epoch).T
                
        elif kernel=='gauss':
            test_K=Gaussian_Kernals(X,test_X,param)
            K_long_test=np.tile(test_K.T,n_epoch).T
                
        else:
            K_long_test=test_X
       
 
    return K_long, Y_one_hot_long, K_long_val, Y_one_hot_val, K_long_test, Y_one_hot_test

def perceptron_fulldata(X_tr, Y_tr, X_val, Y_val, X_test, Y_test, Param):

	'''Run on the full dataset, compute Kernel in loop'''

	k_class=10
	#Matrix of alpha which each row represent each class, change column value after each iteration
	W=np.zeros((k_class,len(X_tr))) 
	Train_error=[]
	Val_error=[]
	Y_pred_test_all=[]
	train_error=0
	val_error=0
	test_error=0

	for t in np.arange(1,len(X_tr),1):
		K=np.zeros((len(X_tr),1))
		#         K[:t-1]=X_tr[:t-1,t] #change here if want to evaluate kernel in loop

		K[:t-1]=Poly_Kernals(X_tr[:t-1],X_tr[t].reshape(1,256),d=Param)
		WX=np.einsum('ij,jk->ik',W,K)
		Y_hat = np.zeros_like(WX) 
		Y_hat[WX.argmax()] = 1 #choosing the class with the highest value
		diff=Y_tr[t]-Y_hat.T[0] 
		train_error+=abs(diff).sum()
		W[:,t]=diff
		Train_error.append(train_error/(t*10))
        
        if X_val is not None:
#             Y_pred_val=np.einsum('ij,jk->ik',W[:,:t], X_val[:t]).T
            for j in range(len(X_val)):
                Y_pred_val=np.einsum('ij,jk->ik',W[:,:t], Poly_Kernals(X_tr[:t],X_val[t].reshape(1,256),d=Param)).T
                B=np.zeros_like(Y_pred_val)
                B[range(len(Y_pred_val)),Y_pred_val.argmax(1)]=1       
                val_error+=abs(B-Y_val).sum()
                Val_error.append(val_error/(len(Y_val)*10)) 
                val_error=0
            
	if X_test is not None:
		#         WX_test=np.einsum('ij,jk->ik',W, X_test).T
		for j in range(len(X_test)):
			WX_test=np.einsum('ij,jk->ik',W[:,:t], Poly_Kernals(X_tr[:t],X_test[j].reshape(1,256),d=Param)).T
			Y_pred_test=np.zeros_like(WX_test)
			Y_pred_test[:,WX_test.argmax(1)]=1 
			test_error+=abs(Y_pred_test-Y_test[j]).sum()
			Y_pred_test_all.append(Y_pred_test)

	return Train_error, Val_error, test_error/(len(X_test)*10), Y_pred_test_all

def perceptron_quick(X_tr, Y_tr, X_val, Y_val, X_test, Y_test):

	'''Quick run with kernalised data'''

	k_class=10
	#Matrix of alpha which each row represent each class, change column value after each iteration
	W=np.zeros((k_class,len(X_tr))) 
	Train_error=[]
	Val_error=[]
	Y_pred_test_all=[]
	train_error=0
	val_error=0
	test_error=0

	for t in np.arange(1,len(X_tr),1):

		WX=np.einsum('ij,jk->ik',W[:,:t],X_tr[:t,t].reshape(t,1))
		Y_hat = np.zeros_like(WX) 
		Y_hat[WX.argmax()] = 1 #choosing the class with the highest value
		diff=Y_tr[t]-Y_hat.T[0] 
		train_error+=abs(diff).sum()
		W[:,t]=diff
		Train_error.append(train_error/(t*10))
        
		if X_val is not None:
			Y_pred_val=np.einsum('ij,jk->ik',W[:,:t], X_val[:t]).T
			#             Y_pred_val=np.einsum('ij,jk->ik',W[:,:t], Poly_Kernals(X_tr[:t],X_val[t].reshape(1,256),d=Param)).T
			B=np.zeros_like(Y_pred_val)
			B[range(len(Y_pred_val)),Y_pred_val.argmax(1)]=1       
			val_error+=abs(B-Y_val).sum()
			Val_error.append(val_error/(len(Y_val)*10))
			val_error=0

		if t>100:
			if Train_error[-1] < np.mean(Val_error[20:]): #stop training if val error > train error
				break
            
	if X_test is not None:
		WX_test=np.einsum('ij,jk->ik',W[:,:t+1], X_test[:t+1,:]).T
		Y_pred_test=np.zeros_like(WX_test)
		Y_pred_test[range(len(Y_test)),WX_test.argmax(1)]=1 
		test_error=abs(Y_pred_test-Y_test).sum()
		Y_pred_test_all.append(Y_pred_test)

	return Train_error, Val_error, test_error/(len(X_test)*10), Y_pred_test_all


def fast_training(data, param_set, epoch, run_no, kernel):

    '''Quick run on reduced dataset with 0.8 training data and 0.2 test'''

    Train_E=[]
    Test_E=[]   
    for i in range(run_no):
    	print(i)
        r_train_index, r_test_index=randomly_split_data(data, 0.8)
        r_train_data=data[r_train_index]
        r_test_data=data[r_test_index]

        for d in param_set:
            X_train, Y_train, _, _, X_test, Y_test = kernalise_data(r_train_data, None, r_test_data, n_class=10, 
                                                n_epoch=epoch, param=d, kernel=kernel)


            train_error, _, test_error, _=perceptron_quick(X_train, Y_train, None, None, X_test, Y_test)

            Train_E.append(train_error)
            Test_E.append(test_error) 
                    
    return np.array(Train_E), np.array(Test_E)


def five_fold_cross_val(data, n_epoch, run_no, param_set, kernel):

	'''Quick run 5fold'''

	train_val_index, test_index=randomly_split_data(data, 0.8)
	train_val_data=data[train_val_index,:]
	test_data=data[test_index,:]

	test_e=[]
	for n in range(run_no):
		print(n)
		np.random.shuffle(train_val_index)
		five_fold_index=np.array(np.array_split(train_val_index,5))

		for d in param_set:
			for f in range(len(five_fold_index)):
				error=0
				val_index=five_fold_index[f]
				train_index=np.setdiff1d(train_val_index, five_fold_index[f])

				val_data=data[val_index,:]
				train_data=data[train_index,:]

				X_train, Y_train, X_val, Y_val, X_test, Y_test=kernalise_data(train_data, val_data, test_data, n_class=10, 
				                                    n_epoch=n_epoch, param=d, kernel=kernel)

				Train_error, Val_error, test_error, Y_pred_test_all=perceptron_quick(X_train, Y_train, X_val, Y_val, X_test, Y_test)
				#             five_fold_train_e.append(Train_error)
				#             five_fold_val_e.append(Val_error)
				error+=test_error
			test_e.append(test_error/5)
        
	return test_e

def main():

	full_data=np.loadtxt('zipcombo.dat.txt')
	reduced_data=reduce_dataset(full_data, 1000)



	D=np.arange(1,8,1)

	Train_E, Test_E=fast_training(reduced_data, D, 5, 2, 'poly')

	np.save('Training_error_20', Train_E)
	np.save('Testing_error_20', Test_E)

	Test_e_5fold=five_fold_cross_val(reduced_data, 5, 2, D, 'poly')

	np.save('Test_error_20_5fold', Test_e_5fold)



	C=[0.01,0.1,1,3,5,7,10]

	Train_E_g, Test_E_g=fast_training(reduced_data, C, 5, 20, 'gauss')

	np.save('Training_error_20_gaussian', Train_E_g)
	np.save('Testing_error_20_gaussian', Test_E_g)

	Test_e_5fold_g=five_fold_cross_val(reduced_data, 5, 20, D, 'gauss')

	np.save('Test_error_20_5fold_gaussian', Test_e_5fold_g)



if __name__ == '__main__':
    main()


