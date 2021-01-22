import numpy as np
from numpy.lib.function_base import average
import pandas as pd
import matplotlib.pyplot as plt

#Polinom Orde degree
class Polinom_Regresion():
    #data_x = potition of vertical soil inclinometer, data_y = gradien of soil inclino, def_actual = actual deflection
    def __init__(self, degre, data_x, data_y, def_actual):
        self.degre = degre
        self.data_x = data_x
        self.data_y = data_y
        self.def_actual = def_actual

    def configure_regresion_Coeff(self):
        if (self.degre+1) > len(self.data_x):
            print('check your data input, the degree+1 must < len(data)')
        else:
            Regresi = np.poly1d(np.polyfit(self.data_x, self.data_y, self.degre))
            Coeff = Regresi.c
        #Boundary Condition at x = 0 , a0 = 0
            b = len(Coeff)
            k = []
            for i in range(len(Coeff)):
                Integrate_coeff = Coeff[i]/b
                b = b - 1
                k.append(Integrate_coeff)
            k.append(0)
            k.reverse()
            p = []
        #Set Boundari condition
            pred = 0 #@x=0, p = 0
            for i in range(len(self.data_x)):
                for j in range(len(k)):
                    pred = pred + (k[j]*(self.data_x[i]**j))
                p.append(pred)
                pred = 0
            y_actual = self.def_actual                         
            y_predict = p 
            ybar = np.sum(y_actual)/len(y_actual)
            ssreg = 0
            sstot = 0
            for i in range(len(y_actual)):
                ssreg = ssreg + ((y_actual[i]-y_predict[i])**2)
                sstot = sstot + ((y_actual[i]-ybar)**2)
            # ssreg = np.sum((self.data_y-ypredict)**2)   
            # sstot = np.sum((self.data_y - ybar)**2)    
            R_square = 1 - (ssreg / sstot)            
            return[p,k,R_square,y_actual,y_predict]

    
#=====================================Read Data====================================================================================================
df = pd.read_excel('Regresi_Data.xlsx')
df_posisi = pd.DataFrame(df, columns=['Posisi'])
df_gradienX = pd.DataFrame(df, columns=['Gradien_X'])
df_gradienZ = pd.DataFrame(df, columns=['Gradien_Z'])
df_defleksiAktualX = pd.DataFrame(df, columns=['Defleksi_Aktual_X'])
df_defleksiAktualZ = pd.DataFrame(df, columns=['Defleksi_Aktual_Z']) 

posisi = df_posisi['Posisi'].to_list()
gradienX = df_gradienX['Gradien_X'].to_list()
gradienZ = df_gradienZ['Gradien_Z'].to_list()
defleksiAktualX = df_defleksiAktualX['Defleksi_Aktual_X'].to_list()
defleksiAktualZ = df_defleksiAktualZ['Defleksi_Aktual_Z'].to_list()
#===================================================================================================================================================
def compute_max_eror(y_actual,y_prediction):
    Error = []
    for i in range(len(y_prediction)):
        if y_actual[i] == 0:
            Err = 0
        else:
            Err = (abs(y_actual[i]-y_prediction[i])/y_actual[i])*100
            Error.append(Err)
    return(max(Error))

R = []
deg = []
def_ac = []
def_pred = []
Error_Pol = []
#Compute Rsquare each polinom regresion
for i in range(len(posisi)):
    S = Polinom_Regresion(i,posisi,gradienX,defleksiAktualX).configure_regresion_Coeff()
    R.append(S[2])
    deg.append(i)
    def_pred.append(S[0])
    Err = compute_max_eror(defleksiAktualX,def_pred[i])
    Error_Pol.append(Err)

    print(defleksiAktualX)
    print(def_pred[i])

    plt.subplot(221)
    plt.plot(def_pred[i],posisi, label = 'P_deg{}'.format(i))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel('Potition [mm]')
    plt.ylabel('Prediction of deflection [mm]')

plt.scatter(defleksiAktualX,posisi)
plt.title('Model Prediction')

plt.subplot(222)
plt.plot(deg,R, marker = 'x', label = 'R^2')
for i,j in zip(deg,R):
    plt.annotate(str(round(j,2)),xy=(i,j))
plt.legend()
plt.ylabel('R_square')
plt.xlabel('degree of polynom')
plt.title('R_Square Parameter')

plt.subplot(212)
plt.plot(deg,Error_Pol, label = 'Max_Error',marker ='p')
for i,j in zip(deg,Error_Pol):
    plt.annotate(str(round(j,2)),xy=(i,j))
plt.legend()
plt.ylabel('Max Error [%]')
plt.xlabel('degree of polinom')
plt.tight_layout()

plt.show()








