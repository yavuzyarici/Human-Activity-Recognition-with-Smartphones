import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


#Data Import and preprocess #################################################

train = pd.read_csv('train.csv/train.csv')
train=train.sample(frac = 1)    #shufle 
X_train= train.iloc[:,:-2].values  #7351x561
y_train= train.iloc[:, -1].values #7352




test = pd.read_csv('test.csv/test.csv')
test=test.sample(frac = 1)  #shufle
X_test= test.iloc[:,:-2].values #2946x561
y_test= test.iloc[:, -1].values #2946 


###########################Onehotencode
classes=['LAYING','SITTING', 'STANDING','WALKING' , 'WALKING_DOWNSTAIRS','WALKING_UPSTAIRS']
y_train_onehot = np.zeros((len(y_train), 6))
y_test_onehot = np.zeros((len(y_test), 6))

for i in range(0, 6):
    for j in range(0, len(y_train)):
        if y_train[j] == classes[i]:
            y_train_onehot[j, i] = 1
        else: 
            y_train_onehot[j, i] = 0

for i in range(0, 6):
    for j in range(0, len(y_test)):
        if y_test[j] == classes[i]:
            y_test_onehot[j, i] = 1
        else: 
            y_test_onehot[j, i] = 0
           
y_train=y_train_onehot
y_test=y_test_onehot

#########################################



p=np.size(X_train,1) #feature len
n=np.size(X_train,0)


################################ Feature Scaling
mean=X_train.mean(axis=0)
var=X_train.std(axis=0)
for i in range(np.size(X_train,1)):
     X_train[:,i] = (X_train[:,i] - mean[i]) / var[i]
for i in range(np.size(X_test,1)):
     X_test[:,i] = (X_test[:,i] - mean[i]) / var[i]    
     
     
#####################"#######################
##PCA
sample_cov_mat = (1/n)*X_train.T@X_train
eig_vals, eig_vecs = np.linalg.eig(sample_cov_mat)
eig_vals = np.real(eig_vals)
eig_vecs = (np.real(eig_vecs)) #Type conversion
total_variance = (1/n)*(np.linalg.norm(X_train,ord='fro'))**2
#Graph of total variance explained vs k
total_var_explained_wrt_k = np.zeros((len(eig_vals),1))
total_var_explained = 0
project_X_train = X_train@eig_vecs
temp = list()
limit = 95 # %
for kk in range(len(eig_vals)):
    total_var_explained += (1/n)*(project_X_train[:,kk].T@project_X_train[:,kk]) / total_variance
    total_var_explained_wrt_k[kk] = (total_var_explained)
    if total_var_explained > limit/100:
        temp.append(kk+1)
how_many_eig_vec_required = temp[0]
plt.plot(np.linspace(1,p,p),100*total_var_explained_wrt_k)
plt.title('Nb. of Principal Components (PC) vs. Total Variance Explained (TVE) (%)\n'
          'Nb. of PC where TVE exceeds {0:d}% first time: {1:d}'.format(limit,how_many_eig_vec_required))
plt.xlabel('Number of Principal Components')
plt.ylabel('Total Variance Explained (%)')
plt.plot(how_many_eig_vec_required,100*total_var_explained_wrt_k[how_many_eig_vec_required],'rx')
plt.plot(np.linspace(1,p,p),limit*np.ones(p),'r--')

u = eig_vecs[:,:how_many_eig_vec_required]
X_train = X_train@u
X_test = X_test@u
p = how_many_eig_vec_required

#######################################

def confusion_matrix(true, pred):
  confusion = np.zeros((6, 6))
  for i in range(len(true)):
    confusion[true[i]][pred[i]] += 1
    confusion.astype(int)
  return confusion

train_acc_list=[]
val_acc_list=[]
past_cost=0
difference_list=[]



def accuracy(X,y,weight):    
    output = []
    for l in range(0, 6):
        h = sigmoid(weight[:,l], X)
        output.append(h)
    output=np.array(output)
    
    predict=np.argmax(output,axis=0)
    true_label=np.argmax(y,axis=1)
    
    accuracy = 0
    
    for row in range(len(y)):
        if true_label[row]==predict[row]:
            accuracy += 1
                
    accuracy = accuracy/len(X)
    return accuracy,predict

def sigmoid(weight,X):
    y = np.dot( weight,X.T)
    return 1 / (1 + np.exp(-y))





def gradient_descent(X, y, weight, learning_rate,decay):
    iteration=0
    past_cost=-10
    while True:
        iteration=iteration+1
        tr_acc=0
        val_acc=0

        cost=0
        for fold in range(0,cv_fold):

            
            X_val=X[735*fold:735*(fold+1)]
            y_val=y[735*fold:735*(fold+1)]

            X_tr=np.concatenate((X[0:735*(fold)],X[735*(fold+1):]),axis=0)
            y_tr=np.concatenate((y[0:735*(fold)],y[735*(fold+1):]),axis=0)

            
            
            for j in range(0, 6):
               
                
                h = sigmoid(weight[:,j,fold], X_tr)           
                cost=cost+(np.sum(y_tr[:,j]*np.log(h+0.00001) + (1-y_tr[:,j])*np.log(1-h+0.00001))+0.00001)*1/len(X_tr)

                for k in range(0, p):
                    weight[k, j,fold] -= (learning_rate/p)*(1/(1+decay*i)) * (np.sum((h-y_tr[:,j])*X_tr[:, k])+ lam*weight[k, j,fold] )
            

            
            acc,_=accuracy(X_tr,y_tr,weight[:,:,fold])
            tr_acc=tr_acc+acc
            acc,_=accuracy(X_val,y_val,weight[:,:,fold])   
            val_acc=val_acc+ acc  
        print('---------cost----')
        cost=cost/60
        difference= (cost-past_cost)/abs(cost)
        print(difference)
        past_cost=cost
        difference_list.append(difference)
        tr_acc=tr_acc*10  #percent scale
        val_acc=val_acc*10
        train_acc_list.append(tr_acc)
        val_acc_list.append(val_acc)
        print('----------------------')
        print('Iteration= %d'%iteration)
        print('train accuracy=%f'%(tr_acc))
        print('validation accuracy=%f'%(val_acc))    
        

        if difference<0.00007:
            break
    return weight

cv_fold=10
lam=0.05


###############################################################Grid Search
'''
lr_list = [0.002,0.003,0.004,0.005,0.006]
decay_list = [0.1,0.2,0.3,0.4,0.5,0.6]
grid_search_acc = np.zeros((len(lr_list),len(decay_list)))
for lr_idx in range(len(lr_list)):
    for C_idx in range(len(decay_list)):
        learning_rate = lr_list[lr_idx]
        decay = decay_list[C_idx]
        weight = np.zeros([p, 6,cv_fold])
        weight = gradient_descent(X_train, y_train, weight,learning_rate ,decay)
        tes_acc,prediction=accuracy(X_test,y_test,weight[:,:,0])
        grid_search_acc[lr_idx,C_idx] =tes_acc
        print("LR_idx: {0}, C_idx: {1}".format(lr_idx, C_idx))


'''
#################################################################


learning_rate=0.04
decay=0.05



weight = np.zeros([p, 6,cv_fold])
weight = gradient_descent(X_train, y_train, weight,learning_rate ,decay)
tes_acc,prediction=accuracy(X_test,y_test,weight[:,:,0])

        

print('test')
print(tes_acc)

true_label=np.argmax(y_test,axis=1)

##############################Confusion Matrix

confusion = confusion_matrix(true_label, prediction)


######################Precision recall F1 score #########

precision = np.zeros(6)
recall = np.zeros(6)
f1_score = np.zeros(6)

for j in range(6):
    precision[j] = confusion[j,j] / sum(confusion[:,j])
    recall[j] = confusion[j,j] / sum(confusion[j,:])
    f1_score[j] = 2*(precision[j]*recall[j])/(precision[j]+recall[j])
 
print('precision:')
print(precision)

print('recall:')
print(recall)

print('f1_score:')
print(f1_score)

#########################################################





plt.figure();
plt.title('Iteration vs. Train Accuracy (%) and Validation Accuracy (%)')
plt.xlabel("Iteration")
plt.ylabel("Accuracy %")
plt.plot(train_acc_list)
plt.plot(val_acc_list)
plt.legend(['Training Accuracy','Validation Accuracy'])
plt.show()   

difference_list=np.array(difference_list)*100

plt.figure();
plt.title('Change in Loss Function % vs Iteration')
plt.xlabel("Iteration")
plt.ylabel("Change in Loss Function % ")
plt.yscale('log')
plt.hlines(y=0.01, xmin=0, xmax=800, linewidth=2, color='r')
plt.plot(difference_list[1:])
plt.legend(['Change in Loss Function %','Stopping Condition'])
plt.show()   