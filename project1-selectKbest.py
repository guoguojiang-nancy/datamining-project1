#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import datetime

from sklearn.metrics import roc_curve, auc,confusion_matrix
import matplotlib as mpl  
import matplotlib.pyplot as plt


# In[2]:


#放弃使用gridsearch 因为运行时间太长了


# In[3]:


Q1=[]


# In[4]:


Tset = pandas.read_csv('new_data.csv')
Teset = pandas.read_csv('test_set.csv')


# In[5]:


Teset.loc[:,'gameId'].value_counts()


# In[6]:


Teset.loc[:,'creationTime'].value_counts()


# In[7]:


Tset.describe()
#观察结果发现没有缺失值，creationTime和gameId统计个数后发现都不是唯一的，因为seasonid的标准差为零，因此seasonid所有都相同（总是相同或者总是不同的特征可以被去掉）


# In[8]:


Teset.describe()


# In[9]:


Tset.drop(['seasonId'], axis=1)
Teset.drop(['seasonId'], axis=1)
# test = Teset.values
# train = Tset.values

# train_set=train[:,0:18]
# train_label=train[:,19]
# test_set=test[:,0:18]
# test_label=test[:,19]


# In[10]:


train_label= Tset['winner']
train_set = Tset.drop(['winner'],axis=1)
test_label= Teset['winner']
test_set = Teset.drop(['winner'],axis=1)


# In[11]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler((0,1))

Mtrain=scaler.fit_transform(train_set)
Mtrain=pandas.DataFrame(Mtrain)
Mtest=scaler.fit_transform(test_set)
Mtest=pandas.DataFrame(Mtest)
Mtrain.describe()


# In[12]:


from sklearn.feature_selection import SelectKBest, GenericUnivariateSelect, chi2

# SelectKBest
best_features = SelectKBest(score_func=chi2, k=len(Mtrain.columns))
fit = best_features.fit(Mtrain, train_label)
df_scores =  pandas.DataFrame(fit.scores_)
df_columns =  pandas.DataFrame(train_set.columns)
# 合并
df_feature_scores =  pandas.concat([df_columns, df_scores], axis=1)
# 定义列名
df_feature_scores.columns = ['Feature', 'Score']
# 按照score排序
df_feature_scores.sort_values(by='Score', ascending=False)


# In[13]:


train_set = Tset.drop(['gameDuration','gameId','creationTime','firstBlood','winner'],axis=1)
train_label= Tset['winner']
test_set = Teset.drop(['gameDuration','gameId','creationTime','firstBlood','winner'],axis=1)
test_label= Teset['winner']
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler((0,1))

train_set=scaler.fit_transform(train_set)
test_set=scaler.fit_transform(test_set)
train_set=pandas.DataFrame(train_set)                              
test_set=pandas.DataFrame(test_set)
train_set.describe()


# In[14]:


# #bayes
# from sklearn.naive_bayes import GaussianNB
# starttime = datetime.datetime.now()
# cls = GaussianNB()
# cls.fit(train_set,train_label)
# bayes_test_predicted_label=cls.predict(test_set)
# cls_score=accuracy_score(bayes_test_predicted_label,test_label)
# bayesMatrix=confusion_matrix(bayes_test_predicted_label,test_label)
# print('bayesMatrix',bayesMatrix)
# print('cls_score',cls_score)
# endtime = datetime.datetime.now()
# bayestime=(endtime - starttime).seconds
# print (bayestime)


#DT
from sklearn import tree
starttime = datetime.datetime.now()

params={'max_depth':np.linspace(1,1,20)}
bestTree = tree.DecisionTreeClassifier()
#gridTree = GridSearchCV(tree, params,  scoring="f1")
bestTree.fit(train_set, train_label)
#bestTree=gridTree.best_estimator_
dt_test_predicted_label=bestTree.predict(test_set)
tree_score=accuracy_score(dt_test_predicted_label,test_label)
Q1.append(("tree",tree_score))
treeMatrix=confusion_matrix(dt_test_predicted_label,test_label)
print('treeMatrix',treeMatrix)
print('tree_score',tree_score)

endtime = datetime.datetime.now()
DTtime=(endtime - starttime).seconds
print (DTtime)

#SVM
from sklearn.svm import SVC
starttime = datetime.datetime.now()

svc = SVC(gamma=0.01,C=1e3,kernel='rbf')
# params={'C':np.linspace(1,0.01,20),'gamma':np.linspace(0.01,0.01,20),'kernel':('linear', 'poly', 'rbf', 'sigmoid')}
# gridSVC = GridSearchCV(svc, params,  scoring="f1")
# gridSVC.fit(train_set, train_label)
# bestSVC=gridSVC.best_estimator_
bestSVC=svc
bestSVC.fit(train_set, train_label)
svc_test_predicted_label=bestSVC.predict(test_set)
SVC_score=accuracy_score(svc_test_predicted_label,test_label)
svmMatrix=confusion_matrix(svc_test_predicted_label,test_label)
print('svmMatrix',svmMatrix)
Q1.append(("SVC",SVC_score))
# print(gridSVC.best_params_ )
print('SVC_score',SVC_score)

endtime = datetime.datetime.now()
SVMtime=(endtime - starttime).seconds
print (SVMtime)

#KNN

from sklearn.neighbors import KNeighborsClassifier
starttime = datetime.datetime.now()

params={'n_neighbors':range(20)}
knn = KNeighborsClassifier()
# gridKNN = GridSearchCV(knn, params,  scoring="f1")
# gridKNN.fit(train_set, train_label)
# bestKNN=gridKNN.best_estimator_
bestKNN=knn
bestKNN.fit(train_set, train_label)
knn_test_predicted_label=bestKNN.predict(test_set)
KNN_score=accuracy_score(knn_test_predicted_label,test_label)
Q1.append(("KNN",KNN_score))
print('KNN_score',KNN_score)
knnMatrix=confusion_matrix(knn_test_predicted_label,test_label)
print('knnMatrix',knnMatrix)
# print(gridKNN.best_params_)

endtime = datetime.datetime.now()
KNNtime=(endtime - starttime).seconds
print (KNNtime)


#MLP
from sklearn.neural_network import MLPClassifier
starttime = datetime.datetime.now()

mlp = MLPClassifier()
# params={'hidden_layer_sizes':range(70)}
# gridMLP = GridSearchCV(mlp, params,  scoring="f1")
# gridMLP.fit(train_set, train_label)
# bestMLP=gridMLP.best_estimator_
bestMLP=mlp
bestMLP.fit(train_set, train_label)
mlp_test_predicted_label=bestMLP.predict(test_set)
MLP_score=accuracy_score(mlp_test_predicted_label,test_label)
Q1.append(("MLP",MLP_score))
mlpMatrix=confusion_matrix(mlp_test_predicted_label,test_label)
print('mlpMatrix',mlpMatrix)
# print(gridMLP.best_params_ )
print('MLP_score',MLP_score)

endtime = datetime.datetime.now()
MLPtime=(endtime - starttime).seconds
print (MLPtime)


# In[15]:


#logistic
#l1
from sklearn.linear_model import LogisticRegression
starttime = datetime.datetime.now()
LR = LogisticRegression(C=1.0, penalty='l1', tol=0.01,solver='liblinear')
LR.fit(train_set, train_label)
test_predicted_label=LR.predict(test_set)
LR_l1_liblinear_score=accuracy_score(test_predicted_label,test_label)
Q1.append(("LR_l1_liblinear",LR_l1_liblinear_score))
# print(gridMLP.best_params_ )
LR_l1_liblinearMatrix=confusion_matrix(test_predicted_label,test_label)
print('LR_l1_liblinearMatrix',LR_l1_liblinearMatrix)
print('LR_l1_liblinear_score',LR_l1_liblinear_score)

endtime = datetime.datetime.now()
LR_l1_liblineartime=(endtime - starttime).seconds
print (LR_l1_liblineartime)

#logistic

starttime = datetime.datetime.now()
LR = LogisticRegression(C=1.0, penalty='l1', tol=0.01,solver='saga')
LR.fit(train_set, train_label)
test_predicted_label=LR.predict(test_set)
LR_l1_saga_score=accuracy_score(test_predicted_label,test_label)
Q1.append(("LR_l1_saga",LR_l1_saga_score))
# print(gridMLP.best_params_ )
LR_l1_sagaMatrix=confusion_matrix(test_predicted_label,test_label)
print('LR_l1_sagaMatrix',LR_l1_sagaMatrix)
print('LR_l1_saga_score',LR_l1_saga_score)

endtime = datetime.datetime.now()
LR_l1_sagatime=(endtime - starttime).seconds
print (LR_l1_sagatime)

#logistic
#l2

starttime = datetime.datetime.now()
LR = LogisticRegression(C=1.0, penalty='l2', tol=0.01,solver='saga')
LR.fit(train_set, train_label)
LR_test_predicted_label=LR.predict(test_set)
LR_l2_saga_score=accuracy_score(LR_test_predicted_label,test_label)
Q1.append(("LR_l2_saga",LR_l2_saga_score))
# print(gridMLP.best_params_ )
LR_l2_sagaMatrix=confusion_matrix(LR_test_predicted_label,test_label)
print('LR_l2_sagaMatrix',LR_l2_sagaMatrix)
print('LR_l2_saga_score',LR_l2_saga_score)

endtime = datetime.datetime.now()
LR_l2_sagatime=(endtime - starttime).seconds
print (LR_l2_sagatime)

#logistic

starttime = datetime.datetime.now()
LR = LogisticRegression(C=1.0, penalty='l2', tol=0.01,solver='newton-cg')
LR.fit(train_set, train_label)
test_predicted_label=LR.predict(test_set)
LR_l2_newtoncg_score=accuracy_score(test_predicted_label,test_label)
Q1.append(("LR_l2_newtoncg",LR_l2_newtoncg_score))
# print(gridMLP.best_params_ )
LR_l2_newtoncgMatrix=confusion_matrix(test_predicted_label,test_label)
print('LR_l2_newtoncgMatrix',LR_l2_newtoncgMatrix)
print('LR_l2_newtoncg_score',LR_l2_newtoncg_score)

endtime = datetime.datetime.now()
LR_l2_newtoncgtime=(endtime - starttime).seconds
print (LR_l2_newtoncgtime)

#logistic

starttime = datetime.datetime.now()
LR = LogisticRegression(C=1.0, penalty='l2', tol=0.01,solver='sag')
LR.fit(train_set, train_label)
test_predicted_label=LR.predict(test_set)
LR_l2_sag_score=accuracy_score(test_predicted_label,test_label)
Q1.append(("LR_l2_sag",LR_l2_sag_score))
# print(gridMLP.best_params_ )
LR_l2_sagMatrix=confusion_matrix(test_predicted_label,test_label)
print('LR_l2_sagMatrix',LR_l2_sagMatrix)
print('LR_l2_sag_score',LR_l2_sag_score)

endtime = datetime.datetime.now()
LR_l2_sagtime=(endtime - starttime).seconds
print (LR_l2_sagtime)

#logistic

starttime = datetime.datetime.now()
LR = LogisticRegression(C=1.0, penalty='l2', tol=0.01,solver='lbfgs',max_iter=500)
LR.fit(train_set, train_label)
test_predicted_label=LR.predict(test_set)
LR_l2_lbfgs_score=accuracy_score(test_predicted_label,test_label)
Q1.append(("LR_l2_lbfgs",LR_l2_lbfgs_score))
# print(gridMLP.best_params_ )
LR_l2_lbfgsMatrix=confusion_matrix(test_predicted_label,test_label)
print('LR_l2_lbfgsMatrix',LR_l2_lbfgsMatrix)
print('LR_l2_lbfgs_score',LR_l2_lbfgs_score)

endtime = datetime.datetime.now()
LR_l2_lbfgstime=(endtime - starttime).seconds
print (LR_l2_lbfgstime)


# In[16]:


from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators=[
    ('bestSVC',SVC(probability=True)),
    ('bestTree', tree.DecisionTreeClassifier()),
    ('LR', LogisticRegression(C=1.0, penalty='l2', tol=0.01,solver='saga')),
    ('mlp', MLPClassifier())
], voting='soft',weights=[1,1,1,7])

voting_clf.fit(train_set, train_label)
voting_clf.score(test_set, test_label)


# In[17]:


from sklearn.ensemble import VotingClassifier
starttime = datetime.datetime.now()
voting_clf = VotingClassifier(estimators=[
    ('bestSVC',SVC(probability=True)),
    ('knn',  KNeighborsClassifier()),
    ('mlp', MLPClassifier())
], voting='soft',weights=[1,1,4])

voting_clf.fit(train_set, train_label)
voting_clf.score(test_set, test_label)
ensemble_test_predicted_label=voting_clf.predict(test_set)
ensemble_score=accuracy_score(ensemble_test_predicted_label,test_label)
Q1.append(("ensemble",ensemble_score))
# print(gridMLP.best_params_ )
ensembleMatrix=confusion_matrix(ensemble_test_predicted_label,test_label)
print('ensembleMatrix',ensembleMatrix)
print('ensemble_score',ensemble_score)
endtime = datetime.datetime.now()
ensembletime=(endtime - starttime).seconds
print (ensembletime)


# In[19]:


Timetable=[['ensemble',ensembletime],['MLPtime',MLPtime],['KNNtime',KNNtime],['DTtime',DTtime],['SVMtime',SVMtime],['LR_l2_sagatime',LR_l2_sagatime]]
print(Timetable)


# In[20]:



confusionTable=[['ensembleMatrix',ensembleMatrix],['LR_l2_sagaMatrix',LR_l2_sagaMatrix],['mlpMatrix',mlpMatrix],['knnMatrix',knnMatrix],['svmMatrix',svmMatrix],['treeMatrix',treeMatrix]]
print(confusionTable)


# In[21]:


print(Q1)


# In[24]:


from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib as mpl  
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,confusion_matrix
plt.title('ROC of svc, ensemble , mlp,logistic,knn')


false_positive_rate,true_positive_rate,thresholds=roc_curve(test_label, svc_test_predicted_label,pos_label=2)
roc_auc=auc(false_positive_rate, true_positive_rate)
plt.plot(false_positive_rate, true_positive_rate,'red',label='svc:        AUC = %0.4f'% roc_auc)

false_positive_rate,true_positive_rate,thresholds=roc_curve(test_label, mlp_test_predicted_label,pos_label=2)
roc_auc=auc(false_positive_rate, true_positive_rate)
plt.plot(false_positive_rate, true_positive_rate,'green',label='mlp: AUC = %0.4f'% roc_auc)

false_positive_rate,true_positive_rate,thresholds=roc_curve(test_label, LR_test_predicted_label,pos_label=2)
roc_auc=auc(false_positive_rate, true_positive_rate)
plt.plot(false_positive_rate, true_positive_rate,'orange',label='logistic: AUC = %0.4f'% roc_auc)

false_positive_rate,true_positive_rate,thresholds=roc_curve(test_label, knn_test_predicted_label,pos_label=2)
roc_auc=auc(false_positive_rate, true_positive_rate)
plt.plot(false_positive_rate, true_positive_rate,'purple',label='knn: AUC = %0.4f'% roc_auc)


   # plt.plot([0,1],[0,1],'bn++')

false_positive_rate,true_positive_rate,thresholds=roc_curve(test_label, dt_test_predicted_label,pos_label=2)
roc_auc=auc(false_positive_rate, true_positive_rate)
plt.plot(false_positive_rate, true_positive_rate,'b',label='ensemble:    AUC = %0.4f'% roc_auc)
plt.plot([0,1],[0,1],'b--')

plt.legend(loc='lower right')
plt.ylabel('TPR')
plt.xlabel('FPR')


# In[ ]:





# In[ ]:




