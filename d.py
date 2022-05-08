import numpy as np
from pandas import DataFrame
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from neupy import algorithms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
dataset = pd.read_csv("studentData/student-por.csv", sep=";")
pd.pandas.set_option('display.max_columns',None)
print(dataset)
sc={
    'GP':1,
    'MS':2,
}
parent={
    'mother':1,
    'father':2,
    'other':3,
}
reas={
    'home':1,
    'reputation':2,
    'course':3,
    'other':4,
}
mjob={
    'teacher':1,
    'health':2,
    'services':3,
    'at_home':4,
    'other':5,

}
fjob={
    'teacher':1,
    'health':2,
    'services':3,
    'at_home':4,
    'other':5,

}
change={
    'yes': 1,
    'no': 0,
}

dataset['address'].replace(to_replace="U", value=1, inplace=True)
dataset['address'].replace(to_replace="R", value=2, inplace=True)
dataset['famsize'].replace(to_replace="LE3", value=1, inplace=True)
dataset['famsize'].replace(to_replace="GT3", value=2, inplace=True)
dataset['Pstatus'].replace(to_replace="T", value=1, inplace=True)
dataset['Pstatus'].replace(to_replace="A", value=2, inplace=True)
dataset['romantic']=dataset['romantic'].map(change)
dataset['internet']=dataset['internet'].map(change)
dataset['famsup']=dataset['famsup'].map(change)
dataset['schoolsup']=dataset['schoolsup'].map(change)
dataset['sex'].replace(to_replace="M",value=1,inplace=True)
dataset['sex'].replace(to_replace="F",value=2,inplace=True)
dataset['Mjob']=dataset['Mjob'].map(mjob)
dataset['Fjob'] = dataset['Fjob'].map(fjob)
dataset['activities']=dataset['activities'].map(change)
dataset['paid']=dataset['paid'].map(change)
dataset['nursery']=dataset['nursery'].map(change)
dataset['higher']=dataset['higher'].map(change)
dataset['reason']=dataset['reason'].map(reas)
dataset['guardian']=dataset['guardian'].map(parent)
dataset['school']=dataset['school'].map(sc)
grade = []
for i in dataset['G3'].values:
    if i in range(0, 10):
        grade.append(4)
    elif i in range(10, 12):
        grade.append(3)
    elif i in range(12, 14):
        grade.append(2)
    elif i in range(14, 16):
        grade.append(1)
    else:
        grade.append(0)

Data1 = dataset
se = pd.Series(grade)
Data1['Grade'] = se.values
dataset.drop(dataset[dataset.G1 == 0].index, inplace=True)
dataset.drop(dataset[dataset.G3 == 0].index, inplace=True)
d1 = dataset
print(d1['failures'])
d1['All_Sup'] = d1['famsup'] & d1['schoolsup']

def max_parenteducation(d1):
    return(max(d1['Medu'],d1['Fedu']))


d1['maxparent_edu']=d1.apply(lambda row: max_parenteducation(row), axis = 1)
#d1['PairEdu'] = d1[['Fedu', 'Medu']].mean(axis=1)
d1['more_high'] = d1['higher'] & (d1['schoolsup'] | d1['paid'])
d1['All_alc'] = d1['Walc'] + d1['Dalc']
d1['Dalc_per_week'] = d1['Dalc'] / d1['All_alc']
d1.drop(['Dalc'], axis=1, inplace=True)
d1.drop(['Walc'], axis=1, inplace=True)
d1['studytime_ratio'] = d1['studytime'] / (d1[['studytime', 'traveltime', 'freetime']].sum(axis=1))
d1.drop(['studytime'], axis=1, inplace=True)
d1.drop(['Fedu'], axis=1, inplace=True)
d1.drop(['Medu'], axis=1, inplace=True)
X = d1.iloc[:, [1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34]]
Y = d1.iloc[:,[28]]
#X = X.astype('float32')
'''scaler.fit(X)
x2 = scaler.transform(X)'''
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.30, random_state=42)
Lvq_net = algorithms.LVQ21(n_inputs=30, n_classes=5, verbose=False, step=0.01, shuffle_data=False)
Lvq_net.train(xTrain, yTrain, epochs=50)
y_training = Lvq_net.predict(xTrain)
y_prediction = Lvq_net.predict(xTest)
print('Prediction accuracy of train data : ')
print('{:.2%}\n'.format(metrics.accuracy_score(yTrain , y_training)))
print('Prediction accuracy of test data : ')
print('{:.2%}\n'.format(metrics.accuracy_score(yTest , y_prediction)))
import pickle
#pd.to_pickle(Lvq_net,r'C:\Users\Gopi Mishra\Desktop\JACOB\PROJECT\OFFICIAL\dlab-main\dlab-main\lvq_ml_model1.pickle')
Lvq_net  = pd.read_pickle(r'C:\Users\Gopi Mishra\PycharmProjects\Main\lvq_ml_model1.pickle')


age_form = float(input("Enter age18-22: "))
gender_form = float(input("Enter Gender:m=1f=2"))
meducation_form = float(input("Enter meducation_form:m=1f=2"))
feducation_form = float(input("Enter feducation_form:m=1f=2"))
mjob_form = float(input("Enter mjob_form:m=1f=2"))
fjob_form = float(input("Enter fjob_form:m=1f=2"))
reason_form = float(input("Enter reason_form:m=1f=2"))
guardian_form = float(input("Enter guardian_form:m=1f=2"))
traveltime_form = float(input("Enter traveltime_form:m=1f=2"))
studytime_form = float(input("Enter studytime_form:m=1f=2"))
failure_form = float(input("Enter failure_form:m=1f=2"))
schoolsup_form = float(input("Enter schoolsup_form:m=1f=2"))
famsup_form = float(input("Enter famsup_form:m=1f=2"))
paid_form = float(input("Enter paid_form:m=1f=2"))
activities_form = float(input("Enter activities_form:m=1f=2"))
nursery_form = float(input("Enter nursery_form:m=1f=2"))
higher_form = float(input("Enter higher_form:m=1f=2"))
internet_form = float(input("Enter internet_form:m=1f=2"))
romantic_form = float(input("Enter romantic_form:m=1f=2"))
famrel_form = float(input("Enter famrel_form:m=1f=2"))
freetime_form = float(input("Enter freetime_form:m=1f=2"))
goout_form = float(input("Enter goout_form:m=1f=2"))
Dalc_form = float(input("Enter Dalc_form:m=1f=2"))
Walc_form = float(input("Enter Walc_form:m=1f=2"))
health_form = float(input("Enter health_form:m=1f=2"))
absences_form = float(input("Enter absences_form:m=1f=2"))
G1_form = float(input("Enter G1_form:m=1f=2"))
G2_form = float(input("Enter G2_form:m=1f=2"))
kk = [G1_form, G2_form]
import statistics
G3_form = statistics.mean(kk)
kk1=[G1_form, G2_form, G3_form]
G3_form1 = statistics.mean(kk1)
All_sup_form  = int(famsup_form) & int(schoolsup_form)
g = max(meducation_form, feducation_form)
maxparent_form = g
morehigh_form = int(higher_form) & (int(schoolsup_form) | int(paid_form))
All_alc_form = Walc_form + Dalc_form
Dalc_week_form = Dalc_form / All_alc_form
k = studytime_form+freetime_form+traveltime_form
studytimeration_form = studytime_form / k
import numpy as np
xyze = np.array([gender_form,age_form,mjob_form,fjob_form,reason_form,guardian_form,traveltime_form,failure_form,schoolsup_form,famsup_form,paid_form,activities_form,nursery_form,higher_form,internet_form,romantic_form,famrel_form,freetime_form,goout_form,health_form,absences_form,G1_form,G2_form,G3_form1,All_sup_form,maxparent_form,morehigh_form,All_alc_form,Dalc_week_form,studytimeration_form], dtype=object)
print(xyze)
result = Lvq_net.predict([xyze])
print(result)