# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:22:35 2021

@author: Mavi
"""
from statsmodels.api import add_constant
#TABLO
from PyQt5.QtWidgets import QApplication,QTableWidgetItem, QMainWindow, QWidget, QInputDialog, QLineEdit, QFileDialog, QLabel, QTextEdit,QGridLayout
import sys
import tasarimUI
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

#VRİSETİ İŞLEMLERİ
from PyQt5.QtWidgets import*
from PyQt5.QtCore import pyqtSlot
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtWidgets import QTextEdit,QLabel,QPushButton,QVBoxLayout,QHBoxLayout,QTableWidget,QTableWidgetItem,QVBoxLayout
from PyQt5.QtGui import QIcon
import os
import matplotlib.pyplot as plt
import pandas as pd

#EPİLEPTİC
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import statsmodels.formula.api as smf
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier#gerekli kütüphaneyi import ettik
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import sklearn.metrics as metrik
from sklearn.svm import LinearSVC
lsvm=LinearSVC()
import pandas as pd
from sklearn import svm 
from sklearn.metrics import accuracy_score
import warnings
import re
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from scipy import stats
import statsmodels.formula.api as smf
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from numpy import mean, absolute
from sklearn.model_selection import GridSearchCV




import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from keras import backend as K
import random
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from sklearn.datasets import load_wine
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import warnings
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import numpy as np





class Pencere(QMainWindow, tasarimUI.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.yukle)
        self.pushButton_2.clicked.connect(self.yukle1)
        self.pushButton_3.clicked.connect(self.yukle2)
        self.pushButton_4.clicked.connect(self.yukle3)
        self.pushButton_5.clicked.connect(self.yukle4)
        self.pushButton_6.clicked.connect(self.yukle5)
        self.pushButton_7.clicked.connect(self.yukle6)
        self.pushButton_8.clicked.connect(self.yukle7)
        self.pushButton_9.clicked.connect(self.yukle8)
        self.pushButton_10.clicked.connect(self.yukle9)
        self.pushButton_12.clicked.connect(self.yukle10)
        self.pushButton_13.clicked.connect(self.yukle11)
        self.pushButton_14.clicked.connect(self.yukle12)
        self.pushButton_15.clicked.connect(self.yukle13)
        self.pushButton_11.clicked.connect(self.yukle14)
        self.pushButton_16.clicked.connect(self.yukle15)
        self.pushButton_17.clicked.connect(self.yukle16)
        self.pushButton_18.clicked.connect(self.yukle17)
        self.pushButton_19.clicked.connect(self.yukle18)
        self.pushButton_20.clicked.connect(self.yukle19)
        self.pushButton_21.clicked.connect(self.yukle20)
        self.pushButton_22.clicked.connect(self.yukle21)
        self.pushButton_23.clicked.connect(self.yukle22)
        self.pushButton_24.clicked.connect(self.yukle23)
        self.pushButton_25.clicked.connect(self.yukle24)
        self.pushButton_26.clicked.connect(self.yukle25)
        self.pushButton_27.clicked.connect(self.yukle26)
        self.pushButton_28.clicked.connect(self.yukle27)
        self.pushButton_29.clicked.connect(self.yukle28)
        
        
        
    #VERİSETİ TABLODA GÖSTR    
    def yukle(self):
         file,_ = QFileDialog.getOpenFileName(self, 'Open file', './',"CSV files (*.csv)") #browse edip dosyayı import etmek için pencere
         self.dataset_file_path = file
         print(self.dataset_file_path)
         self.dataset = pd.read_csv(self.dataset_file_path, engine='python')
         self.dataset = self.dataset.values
        
         print("yükleme=",len(self.dataset[0])) # Öz nitelik sayısı 
         self.tableWidget.clear()
         self.tableWidget.setColumnCount(len(self.dataset[0]))
         self.tableWidget.setRowCount(len(self.dataset))
         for i,row in enumerate(self.dataset):
            for j,cell in enumerate(row):
                self.tableWidget.setItem(i,j, QTableWidgetItem(str(cell)))
         self.tableWidget.horizontalHeader().setStretchLastSection(True)
         self.tableWidget.resizeColumnsToContents() # table gösterim
    

    
    def read_CSV(self,file):  #verisetini okuyup listeye atıyoruz
        with open(file, 'r') as csvFile: # okunabilir dosya
            reader = csv.reader(csvFile)
            for row in reader:
                lines=[]
                for value in row:
                    lines.append(str(round(float(value),3)))
                    
                self.dataset.append(lines)
                
                                    
        csvFile.close()
        
        
        
        
        
    def yukle1(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #df=df.copy()

        #Boş değer varsa dropna ile veriyi sildi. ve ilk 5 satırı head ile gösterdi
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        #print(df.head())
                #verisetini bağımlı bağımsız niteliklere ayırdım
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        df['y']=np.where(df['y'] >1, 0,  1)
        df.y.value_counts()
        self.label_89.setText(str(df.y.value_counts()))
        self.textEdit.setText(str(df))
        self.label.setText(str("1-Nöbet etkinliğinin kaydedilmesi"))
        self.label_2.setText(str("2-Tümörün bulunduğu bölgeden EEG’nin kaydedilmesi"))
        self.label_3.setText(str("3-Tümör bölgesinin beynin neresinde olduğunu belirler ve sağlıklı beyin alanından EEG aktivitesini kaydeder"))
        self.label_4.setText(str("4-Hastanın gözleri kapalıyken EEG sinyalinin kaydedildiği anlamına gelir"))
        self.label_5.setText(str("5-Hastanın gözleri açıkken EEG sinyalinin kaydedildiği anlamına gelir."))



        
        
    #tüm vakaların istatistik bilgilerri   
    def yukle2(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #4.İŞLEM
        #describe() metoduyla verimizin her kolonu için bir takım istatiksel bilgilerine (adet, ortalaması, standart sapması, kolondaki en küçük değer ve en büyük değer, q1, q2, q3 değerleri) ulaşabiliyoruz.
        df.describe().T.head()
        #print(df.describe().T.head())
        self.textEdit_2.setText(str(df.describe().T.head()))
        
        
    def yukle3(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        """df=df.dropna()
        df=df.drop("Unnamed",axis=1)"""
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        df[df['y'] == 0].describe().T
        
        self.textEdit_3.setText(str(df[df['y'] == 0].describe().T))

        
        
        
    def yukle4(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        df[df['y'] == 1].describe().T

        self.textEdit_4.setText(str(df[df['y'] == 1].describe().T))
        
        
    def yukle5(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        
        """# normalizasyondan önce veriyi dengeli hale getir
        import  imblearn
        oversample = imblearn.over_sampling.RandomOverSampler(sampling_strategy='minority')
        # fit and apply the transform
        X, y = oversample.fit_resample(df.drop('y', axis=1), df['y'])
        X.shape, y.shape
        print('Number of records of Non Epileptic {0} VS Epilepttic {1}'.format(len(y == True), len(y == False)))"""
        rf_tuned = RandomForestClassifier().fit(X_train, y_train)#modelimizi eğitiyoruz
        rf_tuned #modelin aldığı parametreler
        Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100 }, index = X_train.columns)
        Importance
        Importance.sort_values(by = "Importance", axis =0, ascending =True).plot(kind = "barh", color ="red")
        
        plt.xlabel ("Değişken Önem Düzeyleri")
        self.textEdit_5.setText(str(Importance))

        
    #LİNEAR SVM   
    def yukle6(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        df=df.copy()
        #0 -1 KAÇ TANE VE GRAFİK GÖSTERİMİ
        #2.İŞLEM
        # y hedef değişkenimizde değeri 1'den farklı olanları "0" sınıfı olarak setliyoruz yani bu onların epilepsi hastası olmadığı anlamına geliyor
        #0  epilepsi hastası değil 1 epilepsi hastası sayılarını ve grafikte gösterimini gösterdim
        """a=len(df['y'])
        for i in range(a):
            if(df['y'][i] != 1):
                df['y'][i]="0"     
        df['y']
        df['y'].value_counts()
        print(df['y'].value_counts())
        df["y"].value_counts().plot.barh(color='red')#garfik"""
        
        
        
        """#Birkaç Epileptik Değil vaka vakası
        #EPİLEPTİK GRAFİKLER
        [(plt.figure(figsize=(8,4)), plt.title('Not Epileptic'), plt.plot(df[df['y'] == 0].iloc[i][0:-1])) for i in range(5)];

        #Birkaç Epileptik Değil vaka vakası
        [(plt.figure(figsize=(8,4)), plt.title('Epileptic'), plt.plot(df[df['y'] == 1].iloc[i][0:-1])) for i in range(5)];

        #y sütunu olmayan tüm verileri içeren dizilerin listeleri
        not_epileptic = [df[df['y']==0].iloc[:, range(0, len(df.columns)-1)].values]
        epileptic = [df[df['y']==1].iloc[:, range(0, len(df.columns)-1)].values]
        
        #2 boyutlu arsa verilerini sıralamak için 2d göstergeler oluşturup hesaplayacağız;
        def indic(data):
            #Göstergeler farklı olabilir. Bizim durumumuzda sadece minimum ve maksimum değerleri kullanıyoruz
            #Ek olarak, ortalama ve standart veya başka bir gösterge kombinasyonu olabilir
            max = np.max(data, axis=1)
            min = np.min(data, axis=1)
            return max, min

        x1,y1 = indic(not_epileptic)
        x2,y2 = indic(epileptic)

        fig = plt.figure(figsize=(14,6))
        ax1 = fig.add_subplot(111)

        ax1.scatter(x1, y1, s=10, c='b', label='Not Epiliptic')
        ax1.scatter(x2, y2, s=10, c='r', label='Epileptic')
        plt.legend(loc='lower left');
        plt.show()


        #sadece Epileptic
        x,y = indic(df[df['y']==1].iloc[:, range(0, len(df.columns)-1)].values)
        plt.figure(figsize=(14,4))
        plt.title('Epileptic')
        plt.scatter(x, y, c='r');



        #sadece Not Epileptic
        x,y = indic(df[df['y']==0].iloc[:, range(0, len(df.columns)-1)].values)
        plt.figure(figsize=(14,4))
        plt.title('NOT Epileptic')
        plt.scatter(x, y);"""
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        print(df.head())
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        
        
        
        #print("LİNEER SVM")
        lsvm.fit(X_train,y_train)
        ypred=lsvm.predict(X_test)
        y_pred = lsvm.predict(X_test)
        
        """predicted = lsvm.predict(X_test)
        #R2 skoru ve Hassasiyet skoru
        #print(f'R2 score: {r2_score(y_test,y_pred)}')
        
        cm = confusion_matrix(y_test, predicted)
        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        classNames = ['Negative','Positive']
        plt.title('SVM LİNEAR Kernel Confusion Matrix - Test Data')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        s = [['TN','FP'], ['FN', 'TP']]

        for i in range(2):
            for j in range(2):
                plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
        plt.show()"""
        
        """print("ROC??????")
        #ROC GRAFİĞİ
        y_train_score3 = lsvm.decision_function(X_train)
        y_train_score4 = lsvm.decision_function(X_train)

        y_train_score3 = lsvm.decision_function(X_train)
        y_train_score4 = lsvm.decision_function(X_train)
        
        fpr,tpr,threshold=metrics.roc_curve(y_train, y_train_score3)
        fpr,tpr,threshold=metrics.roc_curve(y_train, y_train_score4)

        false_pos_rate3, true_pos_rate3, _ = roc_curve(y_train, y_train_score3)
        roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

        false_pos_rate4, true_pos_rate4, _ = roc_curve(y_train, y_train_score4)
        roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14,6))
        ax1.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
        ax1.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
        ax1.set_title('Training Data')
        
        ax1.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc3)
        x1.plot(fpr,tpr,'r',label='AUC=%0.2f'%roc_auc4)

        y_test_score3 = lsvm.decision_function(X_test)
        y_test_score4 = lsvm.decision_function(X_test)

        false_pos_rate3, true_pos_rate3, _ = roc_curve(y_test, y_test_score3)
        roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

        false_pos_rate4, true_pos_rate4, _ = roc_curve(y_test, y_test_score4)
        roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

        ax2.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
        ax2.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
        ax2.set_title('Test Data')
        
        ax2.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc3)
        ax2.plot(fpr,tpr,'r',label='AUC=%0.2f'%roc_auc4)

        for ax in fig.axes:
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([-0.05, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")"""
        #print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))
        self.textEdit_6.setText(str(metrik.accuracy_score(y_pred=ypred,y_true=y_test)))
        #print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
        self.textEdit_7.setText(str(metrik.confusion_matrix(y_pred=ypred,y_true=y_test)))
        acc_linear_svc = round(lsvm.score(X_train, y_train) * 100, 2)
        #print (str(acc_linear_svc) + '%')
        self.textEdit_8.setText(str(acc_linear_svc) + '%')
        #print(classification_report(y_test,y_pred))
        self.textEdit_9.setText(str(classification_report(y_test,y_pred)))
        
        
    #POLİNOM SVM    
    def yukle7(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        #df=df.copy()
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        #df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        #print(df.head())
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        
        print("POLİNOM SVM")
        #SVM Modeli
        model= svm.SVC(kernel='poly',degree=0,gamma='auto_deprecated').fit(X_train,y_train)
        y_pred=model.predict(X_test)
        ypred=model.predict(X_test)
        """predicted = model.predict(X_test)
        #R2 skoru ve Hassasiyet skoru
        #print(f'R2 score: {r2_score(y_test,y_pred)}')
        
        cm = confusion_matrix(y_test, predicted)
        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        classNames = ['Negative','Positive']
        plt.title('SVM POLY Kernel Confusion Matrix - Test Data')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        s = [['TN','FP'], ['FN', 'TP']]

        for i in range(2):
            for j in range(2):
                plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
        plt.show()"""
        
        
        print("ROC??????")
        """ #ROC GRAFİĞİ
        y_train_score3 = model.decision_function(X_train)
        y_train_score4 = model.decision_function(X_train)

        y_train_score3 = model.decision_function(X_train)
        y_train_score4 = model.decision_function(X_train)
        
        fpr,tpr,threshold=metrics.roc_curve(y_train, y_train_score3)
        fpr,tpr,threshold=metrics.roc_curve(y_train, y_train_score4)

        false_pos_rate3, true_pos_rate3, _ = roc_curve(y_train, y_train_score3)
        roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

        false_pos_rate4, true_pos_rate4, _ = roc_curve(y_train, y_train_score4)
        roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14,6))
        ax1.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
        ax1.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
        ax1.set_title('Training Data')
        
        ax1.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc3)
        ax1.plot(fpr,tpr,'r',label='AUC=%0.2f'%roc_auc4)

        y_test_score3 = model.decision_function(X_test)
        y_test_score4 = model.decision_function(X_test)

        false_pos_rate3, true_pos_rate3, _ = roc_curve(y_test, y_test_score3)
        roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

        false_pos_rate4, true_pos_rate4, _ = roc_curve(y_test, y_test_score4)
        roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

        ax2.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
        ax2.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
        ax2.set_title('Test Data')
        
        ax2.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc3)
        ax2.plot(fpr,tpr,'r',label='AUC=%0.2f'%roc_auc4)

        for ax in fig.axes:
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([-0.05, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")"""
        
        
        
        
        
        print(f'Accuracy score: {accuracy_score(y_test,y_pred)}')
        print(classification_report(y_test,y_pred))
        print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
        
        self.textEdit_10.setText(str(f'Accuracy score: {accuracy_score(y_test,y_pred)}'))
        self.textEdit_11.setText(str(classification_report(y_test,y_pred)))
        self.textEdit_12.setText(str(metrik.confusion_matrix(y_pred=ypred,y_true=y_test)))
        
        
    #GUASSİON SVM(RBF)    
    def yukle8(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        #df=df.copy()
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        #df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        #print(df.head())
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        
        # Create a SVC classifier using an RBF kernel
        rsvm =svm.SVC(kernel='rbf', random_state=1, gamma=0.008, C=0.1)
        # Train the classifier
        rsvm.fit(X_train, y_train)
        ypred=rsvm.predict(X_test)
        y_pred = rsvm.predict(X_test)
        
        predicted = rsvm.predict(X_test)
        cm = confusion_matrix(y_test, predicted)
        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        classNames = ['Negative','Positive']
        plt.title('SVM GUASS(RBF) Kernel Confusion Matrix - Test Data')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        s = [['TN','FP'], ['FN', 'TP']]

        for i in range(2):
            for j in range(2):
                plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
        plt.show()
        
        print("ROC??????")
        """#ROC GRAFİĞİ
        y_train_score3 = rsvm.decision_function(X_train)
        y_train_score4 = rsvm.decision_function(X_train)

        y_train_score3 = rsvm.decision_function(X_train)
        y_train_score4 = rsvm.decision_function(X_train)

        fpr,tpr,threshold=metrics.roc_curve(y_train, y_train_score3)
        fpr,tpr,threshold=metrics.roc_curve(y_train, y_train_score4)

        false_pos_rate3, true_pos_rate3, _ = roc_curve(y_train, y_train_score3)
        roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

        false_pos_rate4, true_pos_rate4, _ = roc_curve(y_train, y_train_score4)
        roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14,6))
        ax1.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
        ax1.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
        ax1.set_title('Training Data')
        
        ax1.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc3)
        ax1.plot(fpr,tpr,'r',label='AUC=%0.2f'%roc_auc4)

        y_test_score3 = rsvm.decision_function(X_test)
        y_test_score4 = rsvm.decision_function(X_test)

        false_pos_rate3, true_pos_rate3, _ = roc_curve(y_test, y_test_score3)
        roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

        false_pos_rate4, true_pos_rate4, _ = roc_curve(y_test, y_test_score4)
        roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

        ax2.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
        ax2.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
        ax2.set_title('Test Data')
        
        ax2.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc3)
        ax2.plot(fpr,tpr,'r',label='AUC=%0.2f'%roc_auc4)

        for ax in fig.axes:
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([-0.05, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")"""
        
        print(f'Accuracy score: {accuracy_score(y_test,y_pred)}')
        print(classification_report(y_test,y_pred))
        print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
        
        self.textEdit_14.setText(str(f'Accuracy score: {accuracy_score(y_test,y_pred)}'))
        self.textEdit_13.setText(str(classification_report(y_test,y_pred)))
        self.textEdit_15.setText(str(metrik.confusion_matrix(y_pred=ypred,y_true=y_test)))
        
    #MİN-MAX NORMALİZASYON LİNEAR SVM   
    def yukle9(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        df=df.copy()
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        print(df.head())
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        print("Min-Max   Normalizasyon")
        from sklearn.preprocessing import MinMaxScaler
        mms = MinMaxScaler()
        X_train_normed = mms.fit_transform(X_train) #Eğitim setine normalizasyon uygulamak
        X_test_normed= mms.transform(X_test) #Test setine normalizasyon uygulamak
        print(X_train_normed)
        print(X_test_normed)
        lsvm.fit(X_train_normed,y_train)
        ypred=lsvm.predict(X_test_normed)
        y_pred = lsvm.predict(X_test_normed)
        
        print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))
        self.textEdit_17.setText(str(metrik.accuracy_score(y_pred=ypred,y_true=y_test)))
        
        print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
        self.textEdit_19.setText(str(metrik.confusion_matrix(y_pred=ypred,y_true=y_test)))
        
        acc_linear_svc = round(lsvm.score(X_train_normed, y_train) * 100, 2)
        print (str(acc_linear_svc) + '%')
        self.textEdit_18.setText(str(acc_linear_svc) + '%')
        
        print(classification_report(y_test,y_pred))
        self.textEdit_16.setText(str(classification_report(y_test,y_pred)))
        
        #CONFUSİON MATRİX GRAFİĞİ
        """predicted = lsvm.predict(X_test_normed)
        cm = confusion_matrix(y_test, predicted)
        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        classNames = ['Negative','Positive']
        plt.title('SVM LİNEAR Kernel MİN-MAX- NORM.Confusion Matrix - Test Data')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        s = [['TN','FP'], ['FN', 'TP']]

        for i in range(2):
            for j in range(2):
                plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
        plt.show()"""
        
        """print("ROC??????")
        #ROC GRAFİĞİ
        y_train_score3 = lsvm.decision_function(X_train_normed)
        y_train_score4 = lsvm.decision_function(X_train_normed)

        y_train_score3 = lsvm.decision_function(X_train_normed)
        y_train_score4 = lsvm.decision_function(X_train_normed)
        
        fpr,tpr,threshold=metrics.roc_curve(y_train, y_train_score3)
        fpr,tpr,threshold=metrics.roc_curve(y_train, y_train_score4)

        false_pos_rate3, true_pos_rate3, _ = roc_curve(y_train, y_train_score3)
        roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

        false_pos_rate4, true_pos_rate4, _ = roc_curve(y_train, y_train_score4)
        roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14,6))
        ax1.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
        ax1.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
        ax1.set_title('Training Data')
        
        ax1.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc3)
        x1.plot(fpr,tpr,'r',label='AUC=%0.2f'%roc_auc4)

        y_test_score3 = lsvm.decision_function(X_test_normed)
        y_test_score4 = lsvm.decision_function(X_test_normed)

        false_pos_rate3, true_pos_rate3, _ = roc_curve(y_test, y_test_score3)
        roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

        false_pos_rate4, true_pos_rate4, _ = roc_curve(y_test, y_test_score4)
        roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

        ax2.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
        ax2.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
        ax2.set_title('Test Data')
        
        ax2.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc3)
        ax2.plot(fpr,tpr,'r',label='AUC=%0.2f'%roc_auc4)

        for ax in fig.axes:
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([-0.05, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")"""
            
    
    #STANDART  NORMALİZASYONU KULLANILARAK LİNEER SVM
    def yukle10(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        df=df.copy()
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        print(df.head())
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        
        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        X_train_standart = sc_X.fit_transform(X_train)
        X_test_standart = sc_X.transform(X_test)
        print(X_train_standart)
        print(X_test_standart)
        
        print("STANDART  NORMALİZASYONU KULLANILARAK LİNEER SVM")
        lsvm.fit(X_train_standart,y_train)
        ypred=lsvm.predict(X_test_standart)
        y_pred = lsvm.predict(X_test_standart)
        
        print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))
        self.textEdit_25.setText(str(metrik.accuracy_score(y_pred=ypred,y_true=y_test)))
        
        print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
        self.textEdit_26.setText(str(metrik.confusion_matrix(y_pred=ypred,y_true=y_test)))
        
        acc_linear_svc = round(lsvm.score(X_train_standart, y_train) * 100, 2)
        print (str(acc_linear_svc) + '%')
        self.textEdit_27.setText(str(acc_linear_svc) + '%')
        
        print(classification_report(y_test,y_pred))
        self.textEdit_24.setText(str(classification_report(y_test,y_pred)))
        
        """#CONFUSİON MATRİX GRAFİĞİ
        predicted = lsvm.predict(X_test_standart)
        cm = confusion_matrix(y_test, predicted)
        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        classNames = ['Negative','Positive']
        plt.title('SVM LİNEAR Kernel STANDART NORM.Confusion Matrix - Test Data')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        s = [['TN','FP'], ['FN', 'TP']]

        for i in range(2):
            for j in range(2):
                plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
        plt.show()"""
        
        """print("ROC??????")
        #ROC GRAFİĞİ
        y_train_score3 = lsvm.decision_function(X_train_standart)
        y_train_score4 = lsvm.decision_function(X_train_standart)

        y_train_score3 = lsvm.decision_function(X_train_standart)
        y_train_score4 = lsvm.decision_function(X_train_standart)
        
        fpr,tpr,threshold=metrics.roc_curve(y_train, y_train_score3)
        fpr,tpr,threshold=metrics.roc_curve(y_train, y_train_score4)

        false_pos_rate3, true_pos_rate3, _ = roc_curve(y_train, y_train_score3)
        roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

        false_pos_rate4, true_pos_rate4, _ = roc_curve(y_train, y_train_score4)
        roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14,6))
        ax1.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
        ax1.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
        ax1.set_title('Training Data')
        
        ax1.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc3)
        ax1.plot(fpr,tpr,'r',label='AUC=%0.2f'%roc_auc4)

        y_test_score3 = lsvm.decision_function(X_test_standart)
        y_test_score4 = lsvm.decision_function(X_test_standart)

        false_pos_rate3, true_pos_rate3, _ = roc_curve(y_test, y_test_score3)
        roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

        false_pos_rate4, true_pos_rate4, _ = roc_curve(y_test, y_test_score4)
        roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

        ax2.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
        ax2.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
        ax2.set_title('Test Data')
        
        ax2.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc3)
        ax2.plot(fpr,tpr,'r',label='AUC=%0.2f'%roc_auc4)

        for ax in fig.axes:
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([-0.05, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")"""
            
            
            
    def yukle11(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        df=df.copy()
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        print(df.head())
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
 
        print("Z-Scor NORMALİZASYONU")
        print ("\nZ-skor puanı : \n", stats.zscore(df))
        self.textEdit_28.setText(str(stats.zscore(df)))
        
        
        
    #min-max polinom svm   
    def yukle12(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        df=df.copy()
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        print(df.head())
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        
        
        print("Min-Max  polinom  Normalizasyon")
        from sklearn.preprocessing import MinMaxScaler
        mms = MinMaxScaler()
        X_train_normed = mms.fit_transform(X_train) #Eğitim setine normalizasyon uygulamak
        X_test_normed= mms.transform(X_test) #Test setine normalizasyon uygulamak
        print(X_train_normed)
        print(X_test_normed)
        
        model= svm.SVC(kernel='poly',degree=20,gamma='auto_deprecated').fit(X_train_normed,y_train)
        y_pred=model.predict(X_test_normed)
        ypred=model.predict(X_test_normed)
        
        
 
        print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))
        print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
        print(classification_report(y_test,y_pred))
        

        self.textEdit_30.setText(str(accuracy_score(y_pred=ypred,y_true=y_test)))
        self.textEdit_31.setText(str(confusion_matrix(y_pred=ypred,y_true=y_test)))    
        self.textEdit_29.setText(str(classification_report(y_test,y_pred)))
        
        #CONFUSİON MATRİX GRAFİĞİ
        predicted = model.predict(X_test_normed)
        cm = confusion_matrix(y_test, predicted)
        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        classNames = ['Negative','Positive']
        plt.title('SVM POLİNOM Kernel MİN-MAX NORM.Confusion Matrix - Test Data')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        s = [['TN','FP'], ['FN', 'TP']]

        for i in range(2):
            for j in range(2):
                plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
        plt.show()
        
        """print("ROC??????")
        #ROC GRAFİĞİ
        y_train_score3 = lsvm.decision_function(X_train_normed)
        y_train_score4 = lsvm.decision_function(X_train_normed)

        y_train_score3 = lsvm.decision_function(X_train_normed)
        y_train_score4 = lsvm.decision_function(X_train_normed)
        
        fpr,tpr,threshold=metrics.roc_curve(y_train, y_train_score3)
        fpr,tpr,threshold=metrics.roc_curve(y_train, y_train_score4)

        false_pos_rate3, true_pos_rate3, _ = roc_curve(y_train, y_train_score3)
        roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

        false_pos_rate4, true_pos_rate4, _ = roc_curve(y_train, y_train_score4)
        roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14,6))
        ax1.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
        ax1.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
        ax1.set_title('Training Data')
        
        ax1.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc3)
        ax1.plot(fpr,tpr,'r',label='AUC=%0.2f'%roc_auc4)

        y_test_score3 = lsvm.decision_function(X_test_normed)
        y_test_score4 = lsvm.decision_function(X_test_normed)

        false_pos_rate3, true_pos_rate3, _ = roc_curve(y_test, y_test_score3)
        roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

        false_pos_rate4, true_pos_rate4, _ = roc_curve(y_test, y_test_score4)
        roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

        ax2.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
        ax2.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
        ax2.set_title('Test Data')
        
        ax2.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc3)
        ax2.plot(fpr,tpr,'r',label='AUC=%0.2f'%roc_auc4)

        for ax in fig.axes:
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([-0.05, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")"""
            
            
    def yukle13(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        df=df.copy()
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        print(df.head())
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        print("Standart polinom  Normalizasyon")
        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        X_train_standart = sc_X.fit_transform(X_train)
        X_test_standart = sc_X.transform(X_test)
        print(X_train_standart)
        print(X_test_standart)


        print("POLİNOM SVM")
        #SVM Modeli
        model= svm.SVC(kernel='poly',degree=10,gamma='auto_deprecated').fit(X_train_standart,y_train)
        y_pred=model.predict(X_test_standart)
        ypred=model.predict(X_test_standart)

        
        print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))
        self.textEdit_33.setText(str(metrik.accuracy_score(y_pred=ypred,y_true=y_test)))
        
        print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
        self.textEdit_34.setText(str(metrik.confusion_matrix(y_pred=ypred,y_true=y_test)))
        
        
        print(classification_report(y_test,y_pred))
        self.textEdit_32.setText(str(classification_report(y_test,y_pred)))
        
        #CONFUSİON MATRİX GRAFİĞİ
        predicted = model.predict(X_test_standart)
        cm = confusion_matrix(y_test, predicted)
        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        classNames = ['Negative','Positive']
        plt.title('SVM POLY Kernel STANDART NORM.Confusion Matrix - Test Data')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        s = [['TN','FP'], ['FN', 'TP']]

        for i in range(2):
            for j in range(2):
                plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
        plt.show()
        
        """print("ROC??????")
        #ROC GRAFİĞİ
        y_train_score3 = lsvm.decision_function(X_train_standart)
        y_train_score4 = lsvm.decision_function(X_train_standart)

        y_train_score3 = lsvm.decision_function(X_train_normed)
        y_train_score4 = lsvm.decision_function(X_train_normed)
        
        fpr,tpr,threshold=metrics.roc_curve(y_train, y_train_score3)
        fpr,tpr,threshold=metrics.roc_curve(y_train, y_train_score4)

        false_pos_rate3, true_pos_rate3, _ = roc_curve(y_train, y_train_score3)
        roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

        false_pos_rate4, true_pos_rate4, _ = roc_curve(y_train, y_train_score4)
        roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14,6))
        ax1.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
        ax1.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
        ax1.set_title('Training Data')
        
        ax1.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc3)
        ax1.plot(fpr,tpr,'r',label='AUC=%0.2f'%roc_auc4)

        y_test_score3 = lsvm.decision_function(X_test_standart)
        y_test_score4 = lsvm.decision_function(X_test_standart)

        false_pos_rate3, true_pos_rate3, _ = roc_curve(y_test, y_test_score3)
        roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

        false_pos_rate4, true_pos_rate4, _ = roc_curve(y_test, y_test_score4)
        roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

        ax2.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
        ax2.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
        ax2.set_title('Test Data')
        
        ax2.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc3)
        ax2.plot(fpr,tpr,'r',label='AUC=%0.2f'%roc_auc4)

        for ax in fig.axes:
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([-0.05, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")"""
            
            
            
            
    def yukle14(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        df=df.copy()
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        print(df.head())
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        
        print("Min-Max guassion  Normalizasyon")
        from sklearn.preprocessing import MinMaxScaler
        mms = MinMaxScaler()
        X_train_normed = mms.fit_transform(X_train) #Eğitim setine normalizasyon uygulamak
        X_test_normed= mms.transform(X_test) #Test setine normalizasyon uygulamak
        print(X_train_normed)
        print(X_test_normed)
        
        # Create a SVC classifier using an RBF kernel
        rsvm =svm.SVC(kernel='rbf',random_state=1, gamma=0.008, C=0.1)
        # Train the classifier
        rsvm.fit(X_train_normed, y_train)
        ypred=rsvm.predict(X_test_normed)
        y_pred = rsvm.predict(X_test_normed)
        

        
        self.textEdit_21.setText(str(accuracy_score(y_pred=ypred,y_true=y_test)))
        self.textEdit_22.setText(str(confusion_matrix(y_pred=ypred,y_true=y_test)))    
        self.textEdit_20.setText(str(classification_report(y_test,y_pred)))
        
        """#CONFUSİON MATRİX GRAFİĞİ
        predicted = rsvm.predict(X_test_normed)
        cm = confusion_matrix(y_test, predicted)
        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        classNames = ['Negative','Positive']
        plt.title('SVM GUASSİON(RBF) Kernel MİN-MAX NORM.Confusion Matrix - Test Data')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        s = [['TN','FP'], ['FN', 'TP']]

        for i in range(2):
            for j in range(2):
                plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
        plt.show()"""
        
        
        """print("ROC??????")
        #ROC GRAFİĞİ
        y_train_score3 = rsvm.decision_function(X_train_normed)
        y_train_score4 = rsvm.decision_function(X_train_normed)

        y_train_score3 = rsvm.decision_function(X_train_normed)
        y_train_score4 = rsvm.decision_function(X_train_normed)
        
        fpr,tpr,threshold=metrics.roc_curve(y_train, y_train_score3)
        fpr,tpr,threshold=metrics.roc_curve(y_train, y_train_score4)

        false_pos_rate3, true_pos_rate3, _ = roc_curve(y_train, y_train_score3)
        roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

        false_pos_rate4, true_pos_rate4, _ = roc_curve(y_train, y_train_score4)
        roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14,6))
        ax1.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
        ax1.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
        ax1.set_title('Training Data')
        
        ax1.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc3)
        ax1.plot(fpr,tpr,'r',label='AUC=%0.2f'%roc_auc4)

        y_test_score3 = rsvm.decision_function(X_test_normed)
        y_test_score4 = rsvm.decision_function(X_test_normed)

        false_pos_rate3, true_pos_rate3, _ = roc_curve(y_test, y_test_score3)
        roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

        false_pos_rate4, true_pos_rate4, _ = roc_curve(y_test, y_test_score4)
        roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

        ax2.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
        ax2.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
        ax2.set_title('Test Data')
        
        ax2.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc3)
        ax2.plot(fpr,tpr,'r',label='AUC=%0.2f'%roc_auc4)

        for ax in fig.axes:
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([-0.05, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")"""
            
            
            
    #zkor (standart) guass       
    def yukle15(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        df=df.copy()
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        print(df.head())
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        print("Standart  Normalizasyon")
        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        X_train_standart = sc_X.fit_transform(X_train)
        X_test_standart = sc_X.transform(X_test)
        print(X_train_standart)
        print(X_test_standart)


        # Create a SVC classifier using an RBF kernel
        rsvm =svm.SVC(kernel='rbf', random_state=1, gamma=0.008, C=0.1)
        # Train the classifier
        rsvm.fit(X_train_standart, y_train)
        ypred=rsvm.predict(X_test_standart)
        y_pred = rsvm.predict(X_test_standart)

        
        print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))
        self.textEdit_35.setText(str(metrik.accuracy_score(y_pred=ypred,y_true=y_test)))
        
        print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
        self.textEdit_36.setText(str(metrik.confusion_matrix(y_pred=ypred,y_true=y_test)))
        
        
        print(classification_report(y_test,y_pred))
        self.textEdit_23.setText(str(classification_report(y_test,y_pred)))
        
        """#CONFUSİON MATRİX GRAFİĞİ
        predicted = rsvm.predict(X_test_standart)
        cm = confusion_matrix(y_test, predicted)
        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        classNames = ['Negative','Positive']
        plt.title('SVM GUAASİON(RBF) Kernel STANDART NORM.Confusion Matrix - Test Data')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        s = [['TN','FP'], ['FN', 'TP']]

        for i in range(2):
            for j in range(2):
                plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
        plt.show()"""
        
        """print("ROC??????")
        #ROC GRAFİĞİ
        y_train_score3 = rsvm.decision_function(X_train_standart)
        y_train_score4 = rsvm.decision_function(X_train_standart)

        y_train_score3 = rsvm.decision_function(X_train_standart)
        y_train_score4 = rsvm.decision_function(X_train_standart)
        
        fpr,tpr,threshold=metrics.roc_curve(y_train, y_train_score3)
        fpr,tpr,threshold=metrics.roc_curve(y_train, y_train_score4)

        false_pos_rate3, true_pos_rate3, _ = roc_curve(y_train, y_train_score3)
        roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

        false_pos_rate4, true_pos_rate4, _ = roc_curve(y_train, y_train_score4)
        roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14,6))
        ax1.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
        ax1.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
        ax1.set_title('Training Data')
        
        ax1.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc3)
        ax1.plot(fpr,tpr,'r',label='AUC=%0.2f'%roc_auc4)

        y_test_score3 = rsvm.decision_function(X_test_standart)
        y_test_score4 = rsvm.decision_function(X_test_standart)

        false_pos_rate3, true_pos_rate3, _ = roc_curve(y_test, y_test_score3)
        roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

        false_pos_rate4, true_pos_rate4, _ = roc_curve(y_test, y_test_score4)
        roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

        ax2.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
        ax2.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
        ax2.set_title('Test Data')
        
        ax2.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc3)
        ax2.plot(fpr,tpr,'r',label='AUC=%0.2f'%roc_auc4)

        for ax in fig.axes:
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([-0.05, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")"""
            
            
            
            
    #mad hesaplama        
    def yukle16(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        df=df.copy()
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        #print(df.head())
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        
        # Creating data frame of the given data
        df = pd.DataFrame(df)
        # Absolute mean deviation
        df.mad()# mad() is mean absolute deviation function
        self.textEdit_37.setText(str(df.mad()))
        
        
        
    def yukle17(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #df=df.copy()

        #Boş değer varsa dropna ile veriyi sildi. ve ilk 5 satırı head ile gösterdi
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        #print(df.head())
        #verisetini bağımlı bağımsız niteliklere ayırdım
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        
        #print('Totall Mean VALUE for Epiletic: {}'.format((df[df['y'] == 1].describe().mean()).mean()))
        self.label_92.setText(str('Totall Mean VALUE for Epiletic: {}'.format((df[df['y'] == 1].describe().mean()).mean())))
        #print('Totall Std VALUE for Epiletic: {}'.format((df[df['y'] == 1].describe().std()).std()))
        self.label_93.setText(str('Totall Std VALUE for Epiletic: {}'.format((df[df['y'] == 1].describe().std()).std())))
        
        #print('Totall Mean VALUE for NON Epiletic: {}'.format((df[df['y'] == 0].describe().mean()).mean()))
        self.label_94.setText(str('Totall Mean VALUE for NON Epiletic: {}'.format((df[df['y'] == 0].describe().mean()).mean())))
        
        #print('Totall Std VALUE for NON Epiletic: {}'.format((df[df['y'] == 0].describe().std()).std()))
        self.label_95.setText(str('Totall Std VALUE for NON Epiletic: {}'.format((df[df['y'] == 0].describe().std()).std())))
        
        
        
        
    def yukle18(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        df=df.copy()
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        
        
        #RANDOM FOREST ALGORİTMASI
        from sklearn.ensemble import RandomForestClassifier#gerekli kütüphaneyi import ettik
        rf_model = RandomForestClassifier().fit(X_train, y_train)#modelimizi eğitiyoruz
        rf_model #modelin aldığı parametreler
        RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                               max_depth=None, max_features='auto', max_leaf_nodes=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
                               oob_score=False, random_state=None, verbose=0,warm_start=False)


        y_pred= rf_model.predict(X_test)
        accuracy_score(y_test,y_pred)
        #print(accuracy_score(y_test,y_pred))
        #print(classification_report(y_test,y_pred))
        self.textEdit_38.setText(str(classification_report(y_test,y_pred)))
        self.textEdit_39.setText(str(accuracy_score(y_test,y_pred)))



        rf_params = {"max_depth": [2,5,8,10],"max_features":[2,5,7],"n_estimators":[10,50,100],
                    "min_samples_split":[2,5,10]}
        rf_model =RandomForestClassifier()
        rf_cv_model =GridSearchCV (rf_model, rf_params, cv=10, n_jobs =-1, verbose =5)
        rf_cv_model.fit(X_train, y_train)
        print("En iyi parametreler: "+ str (rf_cv_model.best_params_))
        self.label_99.setText("En iyi parametreler: "+ str (rf_cv_model.best_params_))

        #print("En iyi parametreler: "+ str (rf_cv_model.best_params_)) #en iyi parametreleri bulduk
        rf_tuned = RandomForestClassifier (max_depth = 10, max_features= 2, min_samples_split = 5, n_estimators = 50)
        rf_tuned.fit(X_train, y_train)

        #en iyi parametrelerle tekrar model kuruyoruz
        y_pred= rf_tuned.predict(X_test)
        accuracy_score(y_test,y_pred)
        #print(accuracy_score(y_test,y_pred))
        #print(classification_report(y_test,y_pred))
        self.textEdit_40.setText(str(classification_report(y_test,y_pred)))
        self.textEdit_41.setText(str(accuracy_score(y_test,y_pred)))
        
        
        
    #CNN İLE SINFLANDIRMA    
    def yukle19(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        df=df.copy()
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        #df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        #print(df.head())
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
    
        

        
        
        
        
                  
        def denseBlock(dims,inp) :
            x = tf.keras.layers.BatchNormalization() (inp)
            x = tf.keras.layers.Dense(dims,activation=tf.keras.layers.LeakyReLU(0.2)) (x)
            x = tf.keras.layers.Dropout(0.4) (x)
            x = tf.keras.layers.Dense(dims,activation=tf.keras.layers.LeakyReLU(0.2)) (x)
            x = tf.keras.layers.Dropout(0.4) (x)
            x = tf.keras.layers.Dense(dims,activation=tf.keras.layers.LeakyReLU(0.2)) (x)
            x = tf.keras.layers.Dropout(0.4) (x)
            x = tf.keras.layers.Dense(100,activation=tf.keras.layers.LeakyReLU(0.2)) (x)
            return x

        inp = tf.keras.layers.Input(shape=(178,),name='input')
        x1 = denseBlock(256,inp)
        x2 = denseBlock(512,inp)
        x3 = denseBlock(1024,inp)
        x = tf.keras.layers.Add()([x1,x2,x3])
        x = tf.keras.layers.Dense(128,activation=tf.keras.layers.LeakyReLU(0.2)) (x)
        out = tf.keras.layers.Dense(1,activation='sigmoid',name='output') (x)

        model = tf.keras.models.Model(inp,out)
        model.summary()
        tf.keras.utils.plot_model(model,show_shapes=True)
        model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(1e-4),metrics=['accuracy'])
        model.fit(X_train,y_train,epochs=5,batch_size=128,validation_split=0.2)
        print(model.evaluate(X_test,y_test))
        self.textEdit_42.setText(str("Modelin Test Setinde Değerlendirilmesi: "+ str (model.evaluate(X_test,y_test))))
        
        

                  
    #TRAİN/TEST              
    def yukle20(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        df=df.copy()
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        print(df.head())
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        
        
        """print(X_train.shape)
        print(X_test.shape)
        print(y_test.shape)
        print(y_train.shape)"""
        self.label_102.setText("X-TRAİN: "+ str (X_train.shape))
        self.label_103.setText("X-TEST: "+ str (X_test.shape))
        self.label_104.setText("Y-TEST: "+ str (y_test.shape))
        self.label_105.setText("Y-TRAİN: "+ str (y_train.shape))
        
     
        
    #KNN    
    def yukle21(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        df=df.copy()
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        df.head()
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
         
        
        
        # training a KNN classifier
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)
  
        # accuracy on X_test
        accuracy = knn.score(X_test, y_test)
        #print (accuracy)
        self.textEdit_44.setText(str(accuracy))
  
        # creating a confusion matrix
        knn_predictions = knn.predict(X_test) 
        #print( confusion_matrix(y_test, knn_predictions))
        self.textEdit_45.setText(str(confusion_matrix(y_test, knn_predictions)))

        print(classification_report(y_test,knn_predictions))
        self.textEdit_43.setText(str(classification_report(y_test,knn_predictions)))
        
    #NAVİE BAYES    
    def yukle22(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        df=df.copy()
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        print(df.head())
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        
        # training a Naive Bayes classifier
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB().fit(X_train, y_train)
        gnb_predictions = gnb.predict(X_test)
  
        # accuracy on X_test
        accuracy = gnb.score(X_test, y_test)
        self.textEdit_47.setText(str(accuracy))
  
        # creating a confusion matrix
        confusion_matrix(y_test, gnb_predictions)
        self.textEdit_48.setText(str(confusion_matrix(y_test, gnb_predictions)))
        
        self.textEdit_46.setText(str(classification_report(y_test,gnb_predictions)))
        
    

   #ysa(multi layer(ml))    
    def yukle23(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        df=df.copy()
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        
        from sklearn.neural_network import MLPClassifier
        mlpcl=MLPClassifier(hidden_layer_sizes=(10,10,10,10),max_iter=10000)
        mlpcl.fit(X_train,y_train.values.ravel())
        predictions=mlpcl.predict(X_test)
        from sklearn.metrics import classification_report,confusion_matrix
        #print(confusion_matrix(y_test,predictions))
        self.textEdit_51.setText(str(confusion_matrix(y_test,predictions)))
        
        #print(classification_report(y_test,predictions))
        self.textEdit_49.setText(str(classification_report(y_test,predictions)))
        
        accuracy = mlpcl.score(X_test, y_test)
        #print(accuracy)
        self.textEdit_50.setText(str(accuracy))
        
        
    #multiclass    
    def yukle24(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        df=df.copy()
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
        
        # Loading the dataset
        dataset = load_wine()
        X = dataset.data
        y = dataset.target
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        
        """from sklearn import svm, datasets
        import sklearn.model_selection as model_selection
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import f1_score
        
        rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
        poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)

        poly_pred = poly.predict(X_test)
        rbf_pred = rbf.predict(X_test)

        poly_accuracy = accuracy_score(y_test, poly_pred)
        poly_f1 = f1_score(y_test, poly_pred, average='weighted')
        print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
        print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

        rbf_accuracy = accuracy_score(y_test, rbf_pred)
        rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
        print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
        print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))"""
        
        
        # SVM modelinin oluşturulması
        model = OneVsRestClassifier(SVC())
   
        # Modeli eğitim verileriyle uydurma
        model.fit(X_train, y_train)
   
        # Test seti üzerinde bir tahmin yapmak
        prediction = model.predict(X_test)
   
        # Modeli değerlendirme
        #print(f"Test Set Accuracy : {accuracy_score(y_test, prediction) * 100} %\n\n")
        #print(f"Classification Report : \n\n{classification_report(y_test, prediction)}")
        self.textEdit_54.setText(str(confusion_matrix(y_test,prediction)))
        self.textEdit_53.setText(str(f"Test Set Accuracy : {accuracy_score(y_test, prediction) * 100} %\n\n"))
        self.textEdit_52.setText(str(f"Classification Report : \n\n{classification_report(y_test, prediction)}"))
        
        
        
        
        
    #DWT ayrık dalgacık dönüşümü    
    def yukle25(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        df=df.copy()
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        
        """import pywt
        x = np.linspace(0, 1, num=2048)
        chirp_signal = np.sin(250 * np.pi * x**2)
    
        fig, ax = plt.subplots(figsize=(6,1))
        ax.set_title("Orijinal Chirp Sinyali: ")
        ax.plot(chirp_signal)
        plt.show()
    
        df = chirp_signal
        waveletname = 'sym5'
 
        fig, axarr = plt.subplots(nrows=5, ncols=2, figsize=(6,6))
        for ii in range(5):
           (df, coeff_d) = pywt.dwt(df, waveletname)
           axarr[ii, 0].plot(df, 'r')
           axarr[ii, 1].plot(coeff_d, 'g')
           axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14, rotation=90)
           axarr[ii, 0].set_yticklabels([])
           if ii == 0:
              axarr[ii, 0].set_title("Yaklaşık katsayılar", fontsize=14)
              axarr[ii, 1].set_title("Detay katsayıları", fontsize=14)
           axarr[ii, 1].set_yticklabels([])
        plt.tight_layout()
        plt.show()"""
        
        
        
        
    #eksik verileri saydırdım    
    def yukle26(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        df=df.copy()
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        
        self.textEdit_55.setText(str(df.isnull().sum()))
        
        
        
        
        
    def yukle27(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        df=df.copy()
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        
        
        
        df.isna().any().sum()#kayıp değerler
        
        
        from scipy import signal
        #from scipy.fft import fft, fftfreq
        from scipy.fftpack import fft
        from scipy.fftpack import fftfreq

        # define features, labels 
        features = df.drop(["y"], axis = 1)
        #labels = df["y"]
        # invert the time domain 
        features = features.T
        # switch from time domain to frequency domain with fft 

        # define sampling rate 

        sampling_rate = df.shape[1]

        # remove DC component
        features = features - np.mean(features)

        # fast fourier transformation 
        # fourier_space = [features.iloc[:,i].ravel() for i in range(features.shape[1])]
        # fourier_space = [k for j in fourier_space for k in j]
        """fourier_range = fft(features.T.values.ravel())
        fourier_domain = fftfreq(features.T.values.ravel().size, 1/sampling_rate)

        # use abs to deal with complex numbers 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,5))
        ax1.plot(fourier_domain, np.abs(fourier_range), c = "c")
        ax1.plot(fourier_domain[:fourier_domain.size // 2], np.abs(fourier_range[:fourier_range.size // 2]), c = "g")
        ax1.set_xlabel("frekans [Hz]", fontweight = "bold")
        ax1.set_ylabel("genlik [m]", fontweight = "bold")
        ax2.plot(fourier_domain[:fourier_domain.size // 2], np.abs(fourier_range[:fourier_range.size // 2]), c = "g")
        ax2.set_xlabel("frekans [Hz]", fontweight = "bold")
        ax2.set_ylabel("genlik [m]", fontweight = "bold")
        plt.show()"""
        
        
        
        # Welch's Method 
        # make a plot with log scaling on the y-axis 
        freqs, power_spectrum = signal.welch(features.values.ravel(), sampling_rate, "flattop", 1024, scaling = "spectrum")

        # filter frequency and power spectrum 
        freqs, power_spectrum = freqs[(freqs > 1) & (freqs < 89)], power_spectrum[(freqs > 1) & (freqs < 89)]

        """plt.figure(figsize = (15, 5))
        plt.semilogy(freqs, np.sqrt(power_spectrum), c = "c")
        plt.xlabel("frekans [Hz]", fontweight = "bold")
        plt.ylabel("Linear spectrum [V RMS]", fontweight = "bold")
        plt.show()"""
        # RMS estimate 
        print(f"RMS tahmin: {round(np.sqrt(power_spectrum.max()), 2)}")
        self.label_123.setText(str(f"RMS tahmin: {round(np.sqrt(power_spectrum.max()), 2)}"))
        
        
        # there is a spike in the beta waves, this could be important 
        # majority of spikes in power spectrum is composed of beta waves 
        welch_df = pd.DataFrame({"frekans": freqs, "güç": power_spectrum})
        print(welch_df)
        self.textEdit_56.setText(str(welch_df))
        
        
        
        
        
        
        
        
    def yukle28(self):
        os.chdir('C:/Users/Mavi/Desktop/yapay_zeka_kemalhoca')#username'e kullanici adinizi yaziniz.
        df=pd.read_csv("Epileptic Seizure Recognition.csv")
        #verisetini bağımlı bağımsız niteliklere ayırdım
        df=df.copy()
        df=df.dropna()
        df=df.drop("Unnamed",axis=1)
        #axis=0 satır ,axis=1 kolon demektir yani ona verdiğimiz kolon bazinda silme islemi yapar
        df.head() #verinin ilk 5 satırını ekrana basarak inceleyelim.
        X=df.iloc[:,0:178]
        y=df.iloc[:,178:]
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=824)
        
        
        
       
        #toplam kaç hücrede eksik veri var
        print(df.isnull().sum().sum())
        
        print(df.dtypes)
        
        """from sklearn.decomposition import PCA 
        pca = PCA(n_components=2)
        pca.fit(X)
        x_pca = pca.transform(X)
        print(x_pca)"""
        
    
    
        # Feature Selection with Univariate Statistical Tests

        from numpy import set_printoptions
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_classif
        from sklearn.feature_selection import RFE
        from sklearn.decomposition import PCA
        from sklearn.ensemble import ExtraTreesClassifier

        #en iyi 4 özellik seçilir
        # feature extraction
        test = SelectKBest(score_func=f_classif, k=4)
        fit = test.fit(X, y)
        # summarize scores
        set_printoptions(precision=3)
        print(fit.scores_)
        features = fit.transform(X)
        # summarize selected features
        print(features[0:5,:])
            
        
        """model = LogisticRegression(solver='lbfgs')
        rfe = RFE(model, 3)
        fit = rfe.fit(X, y)
        print("Sayı Özellikleri: %d" % fit.n_features_)
        print("Seçilmiş Özellikler: %s" % fit.support_)
        print("Özellik Sıralaması: %s" % fit.ranking_)"""
        
        
        
        # feature extraction
        pca = PCA(n_components=3)
        fit = pca.fit(X)
        # summarize components
        print("Açıklanan Varyans: %s" % fit.explained_variance_ratio_)
        print(fit.components_)
        
        
        
        # feature extraction
        model = ExtraTreesClassifier(n_estimators=10)
        model.fit(X, y)
        print(model.feature_importances_)
        
        
        
        





        
        
        
    
  
        
        
        
        
        
        
        
        
        


     



        


        

        

app=QApplication(sys.argv)
pencere=Pencere()
pencere.show()
sys.exit(app.exec_())