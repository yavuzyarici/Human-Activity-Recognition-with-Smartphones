#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

#%%
#Data Import and preprocess #################################################

train = pd.read_csv('train.csv/train.csv')
train=train.sample(frac = 1)    #shufle 
X_train= train.iloc[:,:-2].values  #7351x561
y_train= train.iloc[:, -1].values #7352

test = pd.read_csv('test.csv/test.csv')
test=test.sample(frac = 1)  #shufle
X_test= test.iloc[:,:-2].values #2946x561
y_test= test.iloc[:, -1].values #2946 
tr=y_train
tst=y_test


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
del y_train_onehot, y_test_onehot
#########################################

p=np.size(X_train,1) #feature len
n=np.size(X_train,0)
# Feature Scaling

################################ Feature Scaling
mean=X_train.mean(axis=0)
var=X_train.std(axis=0)
for i in range(np.size(X_train,1)):
     X_train[:,i] = (X_train[:,i] - mean[i]) / var[i]
for i in range(np.size(X_test,1)):
     X_test[:,i] = (X_test[:,i] - mean[i]) / var[i]    
#######################################

#####################"#######################
##PCA
usePCA = True
if usePCA:
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
    limit = 90 # %
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
    del project_X_train, eig_vals, eig_vecs, temp

##################################################################################
use_rbf_kernel = False
if use_rbf_kernel:
    gamma = 1/(np.var(X_train)*X_train.shape[1])
    p = newFeatureNb = 250
    new_data_train = np.zeros((n,newFeatureNb))
    new_data_test = np.zeros((X_test.shape[0],newFeatureNb))
    f_idx = 0
    for sample_idx in np.random.choice(range(X_train.shape[0]),newFeatureNb,replace=False):
        print('Sample idx:',sample_idx)
        l = X_train[sample_idx,:]
        distanceSqrSum = np.sum((X_train-l)**2,axis=1)
        new_data_train[:,f_idx] = np.exp(-distanceSqrSum*gamma)
        distanceSqrSum = np.sum((X_test-l)**2,axis=1)
        new_data_test[:,f_idx] = np.exp(-distanceSqrSum*gamma)
        f_idx += 1
    X_train = new_data_train
    X_test= new_data_test
    del new_data_train,new_data_test

feature_scaling_for_rbf = True
if use_rbf_kernel and feature_scaling_for_rbf:
    del mean, var
    mean=X_train.mean(axis=0)
    var=X_train.std(axis=0)
    for i in range(np.size(X_train,1)):
         X_train[:,i] = (X_train[:,i] - mean[i]) / var[i]
    for i in range(np.size(X_test,1)):
         X_test[:,i] = (X_test[:,i] - mean[i]) / var[i]    

##################################################################################
def confusion_matrix(true, pred):
  confusion = np.zeros((6, 6))
  for i in range(len(true)):
    confusion[true[i]][pred[i]] += 1
    confusion.astype(int)
  return confusion

## Hyper Parameters
C = 1000#C for loss of svm
how_many_fold = 10
fold_size = n//how_many_fold
epoch = 100
nb_of_class = y_train.shape[1]
d = 1 #delta is constant chosen as 1, not trying to optimize, C will be optimized instead
lr = 1e-3

#weights is defined as (p x nb_of_class)
#Cost Fnc: C * hinge loss + 1/2 ||w||^2 is used
def calculate_loss(weights, C, X, Y):
    
    nb_of_sample = X.shape[0]
    
    #Finding scores for each class
    Y_idxed = np.reshape(np.argmax(Y,1),-1)
    scores = X @ weights
    scores_of_true_class = scores[Y>0.5]
    marjin = np.maximum(d + scores - scores_of_true_class[:,np.newaxis],0)
    
    loss_margin = np.sum(marjin) / n
    return (C*loss_margin + 0.5*np.sum(weights*weights))

def calculate_grad(weights, C, X, Y):
    
    nb_of_sample = X.shape[0]
    
    #Finding scores for each class
    Y_idxed = np.reshape(np.argmax(Y,1),-1)
    scores = X @ weights
    scores_of_true_class = scores[Y>0.5]
    marjin = np.maximum(d + scores - scores_of_true_class[:,np.newaxis],0)
    
    loss_margin = np.sum(marjin) / n
    
    #We need to determine where the margin is greaterr than 0
    how_many_greater_than_zero_margin = np.sum(marjin >0,axis=1)
    X_modified_for_grad = (marjin>0).astype(float)
    X_modified_for_grad[Y>0.5] = -how_many_greater_than_zero_margin
    grad = C * X.T @ X_modified_for_grad / n + weights
    return grad


def accuracy(X,y,weights):    
    estimates = np.argmax(X @ weights,axis=1)
    real = np.argmax(y,axis=1)
    true_labeled = np.sum(real==estimates)
    return true_labeled/X.shape[0]


#%%
###################################################################
#Training with entire train set for final results
initial_time = time.time() 
weights = np.random.randn(X_train.shape[1],y_train.shape[1])
for epoch_nb in range(1,1+epoch):
    weights -= lr*calculate_grad(weights, C, X_train, y_train)
elapsed = time.time()-initial_time
print("Elapsed Time: {0:.5f}".format(elapsed))
print(accuracy(X_test,y_test,weights))


#%%
###################################################################
## Cross Validational Training for epoch based graph

train_acc_list=[]
val_acc_list=[]

#Shuffling the data
order_of_data = np.linspace(0,n-1,n,dtype=int)
np.random.shuffle(order_of_data)
shuffled_X_train = X_train[order_of_data,:]
shuffled_y_train = y_train[order_of_data,:]
for fold_nb in range(1,how_many_fold+1):
    train_acc_fold = list()
    val_acc_fold = list()
    #Divide validational fold and training part
    fold_train_x = np.concatenate((shuffled_X_train[:(fold_nb-1)*fold_size,:],shuffled_X_train[fold_nb*fold_size:,:]))
    fold_train_y = np.concatenate((shuffled_y_train[:(fold_nb-1)*fold_size,:],shuffled_y_train[fold_nb*fold_size:,:]))
    fold_val_x = shuffled_X_train[(fold_nb-1)*fold_size:fold_nb*fold_size-1,:]
    fold_val_y = shuffled_y_train[(fold_nb-1)*fold_size:fold_nb*fold_size-1,:]
    weights = np.random.randn(fold_val_x.shape[1],fold_val_y.shape[1])
    for epoch_nb in range(1,1+epoch):
        weights -= lr*calculate_grad(weights, C, fold_train_x, fold_train_y)
        val_acccc = accuracy(fold_val_x,fold_val_y,weights)
        train_acc_fold.append(accuracy(fold_train_x,fold_train_y,weights))
        val_acc_fold.append(val_acccc)
        print("Fold {0}, epoch {1} val accuracy: {2:.3f}".format(fold_nb, epoch_nb,val_acccc))
    train_acc_list.append(train_acc_fold)
    val_acc_list.append(val_acc_fold)


test_acc = accuracy(X_test,y_test,weights)

print('test')
print(test_acc)

# true_label=np.argmax(y_test,axis=1)
# confusion = confusion_matrix(true_label, prediction)

# plot_confusion_matrix(cm=confusion, target_names=['LAYING','SITTING', 'STANDING','WALKING' , 'WALKING_DOWNSTAIRS','WALKING_UPSTAIRS' ], title='Confusion Matrix')

plt.figure();
plt.title('Iteration vs. Train Accuracy (%) and Validation Accuracy (%)\nTest Set Accuracy = {0:.2f}%'.format(100*test_acc))
plt.xlabel("Iteration")
plt.ylabel("Accuracy (%)")


avg_train_acc = np.zeros((len(train_acc_list[0]),1))
for i in range(len(train_acc_list)):
    avg_train_acc += np.reshape(np.array(train_acc_list[i]),(-1,1))
avg_train_acc /= len(train_acc_list)

avg_val_acc = np.zeros((len(val_acc_list[0]),1))
for i in range(len(val_acc_list)):
    avg_val_acc += np.reshape(np.array(val_acc_list[i]),(-1,1))
avg_val_acc /= len(val_acc_list)


plt.plot(100*avg_train_acc)
plt.plot(100*avg_val_acc)
plt.legend(['Training Accuracy','Validation Accuracy'])
plt.show()   


###################################################################
#%%
## Cross Validational Training for Parameter Selection
epoch = 100
train_acc_list=[]
val_acc_list=[]
lr_list = [1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1]
C_list = [1e1,1e2,1e3,1e4,1e5,1e6,1e7]
all_accs = list()
#Shuffling the data
order_of_data = np.linspace(0,n-1,n,dtype=int)
np.random.shuffle(order_of_data)
shuffled_X_train = X_train[order_of_data,:]
shuffled_y_train = y_train[order_of_data,:]
grid_search_acc = np.zeros((len(lr_list),len(C_list)))
for lr_idx in range(len(lr_list)):
    for C_idx in range(len(C_list)):
        lr = lr_list[lr_idx]
        C = C_list[C_idx]
        val_acc= list()
        for fold_nb in range(1,how_many_fold+1):
            #Divide validational fold and training part
            fold_train_x = np.concatenate((shuffled_X_train[:(fold_nb-1)*fold_size,:],shuffled_X_train[fold_nb*fold_size:,:]))
            fold_train_y = np.concatenate((shuffled_y_train[:(fold_nb-1)*fold_size,:],shuffled_y_train[fold_nb*fold_size:,:]))
            fold_val_x = shuffled_X_train[(fold_nb-1)*fold_size:fold_nb*fold_size-1,:]
            fold_val_y = shuffled_y_train[(fold_nb-1)*fold_size:fold_nb*fold_size-1,:]
            weights = np.random.randn(fold_val_x.shape[1],fold_val_y.shape[1])
            temp_acc = list()
            for epoch_nb in range(1,1+epoch):
                weights -= lr*calculate_grad(weights, C, fold_train_x, fold_train_y)
                temp_acc.append(accuracy(fold_val_x, fold_val_y, weights))
            val_acc.append(accuracy(fold_val_x, fold_val_y, weights))
            all_accs.append(temp_acc)
        grid_search_acc[lr_idx,C_idx] = (sum(val_acc)/len(val_acc))
        print("LR_idx: {0}, C_idx: {1}".format(lr_idx, C_idx))
        
        
#Imshow
fig, ax = plt.subplots(1,1)

img = ax.imshow(100*grid_search_acc)

plt.xlabel("C Values")
plt.ylabel("Learning Rates")
plt.title("10 Fold Cross Validational Accuracies (%) with\nDifferent C and Learning Rate Values\nHighest"+
          " Accuracy found with lr = 1e-3, C = 1000")
x_label_list = ["","1e1","1e2","1e3","1e4","1e5","1e6","1e7"]
y_label_list = ["","1e-4","3e-4","1e-3","3e-3","1e-2","3e-2","1e-1"]
ax.set_xticklabels(x_label_list)
ax.set_yticklabels(y_label_list)

fig.colorbar(img)
# lr = 1e-3, C = 1000 selected


###############################################
### Confusion Matrix

prediction = np.argmax(X_test @ weights,axis=1)
true_label=np.argmax(y_test,axis=1)
confusion = confusion_matrix(true_label, prediction)




######################Precision recall F1 score

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
#################################################