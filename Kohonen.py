"""
Universidade Federal Fronteira  Sul
Ciência da Computação
Inteligência Artificial
Ivair Puerari

Problema:
    O trabalho consiste em desenvolver um reconhecedor de sementes
usando a rede neural Kohonen.

"""
import sys
import numpy as np
from minisom import MiniSom
from matplotlib import pyplot as plt
import os.path


#Flag para gerar um arquivo datasuffle_.txt ou randomicamente selecionar um conjunto
#de dados novo sempre.
# 0 - Randomicamente sempre
# 1 - Gera um arquivo datasuffle.txt
SUFFLE_FLAG = 1
#Variavel auxiliar para gerar um arquivo datasuffle_.txt que substituira _ no nome
#do arquivo.
VERSION = 3

#Função que gera um arquivo de conjunto de dados embaralhado
def process():
    if(os.path.isfile("datasuffle"+str(VERSION)+".txt") == False):
        file = np.loadtxt('seeds_dataset.txt')
        np.random.shuffle(file)
        np.savetxt("datasuffle"+str(VERSION)+".txt",file,delimiter=',')
        

#Função que realiza a escala no conjuto de dados
def featureScaling(X):
  Xmean=X-np.mean(X,0)
  stde= np.std(X,0, ddof=1) 
  return Xmean/stde

#Função que realiza a classificação de cada neuronio com a sua devida label
#Plotando os neuronios ativados na classificação
def classify(Xtest,ytest,maps_train):
    error = np.zeros(4)
    labels = np.zeros((nsize,nsize),dtype=int)
    fig, ax = plt.subplots()
    for cnt, xx in enumerate(Xtest):
        
        w = som.winner(xx)
        labels[w] = ytest[cnt]

        if(ytest[cnt] == 1):
            ax.scatter(w[0],w[1],color = 'Red')
            ax.annotate(1,w,color = 'black')
           
               
        if(ytest[cnt] == 2):
            ax.scatter(w[0],w[1],color = 'blue')
            ax.annotate(2,w,color = 'black')
            

         
        if(ytest[cnt] == 3):
            ax.scatter(w[0],w[1],color = 'Yellow')
            ax.annotate(3,w,color = 'black')
           

        if(maps_train[w] != ytest[cnt]):
            error[int(ytest[cnt])] = error[int(ytest[cnt])] +1 
    plt.title('Classe 1 : Kama, Classe 2 : Rosa, Classe 3 : Canadian')
    plt.show()
    return labels,error

#Função que plota o som após realizar o treino da rede
def plot_som(Xplot,yplot):
    labels = np.zeros((nsize,nsize),dtype=int)
    fig, ax = plt.subplots()
    for cnt, xx in enumerate(Xplot):
        
        w = som.winner(xx)
        labels[w] = yplot[cnt]

        if(yplot[cnt] == 1):
            ax.scatter(w[0],w[1],color = 'Red')
            ax.annotate(1,w,color = 'black')
        if(yplot[cnt] == 2):
            ax.scatter(w[0],w[1],color = 'blue')
            ax.annotate(2,w,color = 'black')
        if(yplot[cnt] == 3):
            ax.scatter(w[0],w[1],color = 'Yellow')
            ax.annotate(3,w,color = 'black')
    plt.title('Classe 1 : Kama, Classe 2 : Rosa, Classe 3 : Canadian')

    plt.show()
    return labels

#Função que verifica a quantidade de ocorrencias de classe nos conjuntos treino e teste de dados 
def occurrence(a):
    unique, counts = np.unique(a, return_counts=True)
    d = dict(zip(unique, counts))
    count = []
    for i in d.values():
        count.append(i)
    
    plt.bar(d.keys(), d.values(), color=['red','blue','yellow']) 
    plt.xticks([1,2,3], ('Kama', 'Rosa', 'Canadian'))
    plt.yticks(np.arange(max(count)+1))
    plt.title('Ocorrências para cada classe')
    plt.show()    
    return d 

if __name__ == "__main__":
    
    #O conjunto de dado é ordenado, logo é necessario embaralhar os dados
    # para que, ao dividir em conjunto de treino e teste
    # os dados tenham exemplos de todas classes 
    if(SUFFLE_FLAG == 0):
        try:
            data = np.loadtxt(sys.argv[1])
            np.random.shuffle(data)
        except:
            print('Could not open', sys.argv[1])
            exit(0)
    else:
        try:
            process()
            data = np.loadtxt("datasuffle"+str(VERSION)+".txt",delimiter=',')
        except:
            print('Could not open', sys.argv[1])
            exit(0)
        
    #Pega numero de exemplos e caracteristicas do conjunto de dados
    m = len(data)
    d = len(data[0])
    X = data[:, 0:d-1]
    y = data[:,d-1:d]
   

    
    #APlica a função de escala para valores estarem entre 0 e 1
    X = featureScaling(X)
   
    #Tamanho do conjunto de treino
    tsize=int(m*0.7) 
    
    #Divide o conjunto de dados em dois
    #Conjunto de treino e teste
    Xtr=X[:tsize,:] 
    Xte=X[tsize:,:] 

    ytr=y[:tsize]
    yte=y[tsize:]
    
    #Pega numero de exemplos e caracteristicas do conjunto treino
    m,d = Xtr.shape

    #numero de neuronios  da rede
    nsize = 5 * np.sqrt(m)
    nsize = int(np.sqrt(nsize))-1

    #Constroi e inicializa o modelo SOM
    som = MiniSom(nsize, nsize, d, sigma=0.8, learning_rate=0.2,random_seed=0)
    som.random_weights_init(Xtr)
    
    #Treina a rede
    n_iteration = 1000
    som.train_random(Xtr,n_iteration)

   
    ytr = np.array(ytr).flatten()
    yte =  np.array(yte).flatten()

    #Plota o mapa SOM e retorna o mapa
    maps_train = plot_som(Xtr,ytr)

    #classifica o modelo para o conjunto de teste
    teste,error = classify(Xte,yte,maps_train)

    #verifica as ocorrencias das classes para cada conjunto, treino e teste
    nlabels_tr = occurrence(ytr)
    nlabels_te = occurrence(yte)

    #Realiza os calculos de acurracia, verificando os acertos para cada classe
    labels = ['Kama','Rosa','Canadian']
    mean_acc = np.zeros(3)
    print("\n")
    for i in range(1,4):
        mean_acc[i-1] = (nlabels_te[i] - error[i]) / nlabels_te[i]
        print("Classe {} - {}: {}  ocorrencias com  acerto: {:.2f} %.".format(i,labels[i-1],nlabels_te[i],mean_acc[i-1]*100))
 
    #Imprime a media alcançada pelo modelo
    print("\nMédia de acertos {:.2f} %.\n".format(np.mean(mean_acc)*100))