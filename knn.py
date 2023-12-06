import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mode
import time




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
#######################################

y_test = np.argmax(y_test,axis=1)
y_train = np.argmax(y_train,axis=1)
#Labels are hold as an integer 0 to 6
##################################################################################

def confusion_matrix(true, pred):
  confusion = np.zeros((6, 6))
  for i in range(len(true)):
    confusion[true[i]][pred[i]] += 1
    confusion.astype(int)
  return confusion

## Hyper Parameters
k = 3

def knnRun(X_stored_data,y_stored_data, X_for_prediction, k, metric='l2'):
    output = np.zeros((X_for_prediction.shape[0],1))
    for i in range(X_for_prediction.shape[0]):
        data_point = X_for_prediction[i,:]
        distances = X_stored_data-data_point
        if metric == 'l2':
            euc_distances = np.sum(distances**2, axis = 1)
        else:
            euc_distances = np.sum(np.abs(distances), axis = 1)
        output[i] = int(mode(y_stored_data[euc_distances.argsort()[0:k]])[0][0])
    return output.T
def accuracy(y_true, y_estimate):    
    true_labeled = np.sum(y_true==y_estimate)
    return true_labeled/len(y_true)
#%%
###################################################################

## Cross Validational Training for k - parameter selection
val_acc_list=[]
how_many_fold = 10
fold_size = n//how_many_fold
#Shuffling the data
order_of_data = np.linspace(0,n-1,n,dtype=int)
np.random.shuffle(order_of_data)
shuffled_X_train = X_train[order_of_data,:]
ks = np.arange(1,22,2)
shuffled_y_train = y_train[order_of_data]
val_acc_k_selection = list()
for k in ks:
    accs_for_one_k = list()
    initial_time = time.time()
    for fold_nb in range(1,11):
        #Divide validational fold and training part
        fold_train_x = np.concatenate((shuffled_X_train[:(fold_nb-1)*fold_size,:],shuffled_X_train[fold_nb*fold_size:,:]))
        fold_train_y = np.concatenate((shuffled_y_train[:(fold_nb-1)*fold_size],shuffled_y_train[fold_nb*fold_size:]))
        fold_val_x = shuffled_X_train[(fold_nb-1)*fold_size:fold_nb*fold_size-1,:]
        fold_val_y = shuffled_y_train[(fold_nb-1)*fold_size:fold_nb*fold_size-1]
        y_estimate = knnRun(fold_train_x, fold_train_y, fold_val_x, k)
        accs_for_one_k.append(accuracy(fold_val_y, y_estimate))
        print(k)
        elapsed = time.time()-initial_time
        print("Elapsed Time: {0:.0f}".format(elapsed))
    val_acc_k_selection.append(sum(accs_for_one_k)/len(accs_for_one_k))        

optimal_k = ks[np.argmax(val_acc_k_selection)]

test_acc = accuracy(y_test,knnRun(X_train, y_train, X_test, k))

plt.figure();
plt.title('L2 | k Parameter vs Validation Accuracy (%)\nTest Set Accuracy (with optimal k: {1}) = {0:.2f}%'.format(100*test_acc, optimal_k))
plt.xlabel("k parameter")
plt.ylabel("Accuracy (%)")
plt.plot(ks, 100*np.array(val_acc_k_selection))
#%%

#%%
##### Final time 
initial_time = time.time()
prediction_test_y= np.asarray(knnRun(X_train, y_train, X_test, 9),int)
elapsed = time.time()-initial_time
print("Elapsed Time: {0:.0f}".format(elapsed))
###############################################
### Confusion Matrix
true_label = y_test
confusion = confusion_matrix(true_label, prediction_test_y.T)



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