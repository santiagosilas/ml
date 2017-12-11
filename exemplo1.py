from sklearn import tree

# exemplos de treinamento
tilapia1 = [1, 0.2]
tilapia2 = [3, 0.5]
tilapia3 = [2, 0.7]
salmao1 = [5, 0.9]
salmao2 = [3, 0.8]
salmao3 = [4, 0.6]

# conjunto de treinamento
exemplos = [tilapia1, tilapia2, tilapia3, salmao1, salmao2, salmao3]
rotulos = [1, 1, 1, 2, 2, 2]



# treinamento do modelo
modelo = tree.DecisionTreeClassifier()
modelo.fit(exemplos, rotulos)

# exemplos de teste
peixe1 = [3, 0.5]
peixe2 = [2, 0.6]
peixe3 = [5, 0.3]

# conjunto de treino
teste = [peixe1, peixe2, peixe3]
rotulos_teste = [1, 1, 2]

# predições
predicoes = modelo.predict(teste)

# computar a taxa de acertos
acertos = 0
for rotulo, predicao in zip(rotulos_teste, predicoes):
    print('rotulo:', rotulo, 'predicao:', predicao)
    if rotulo == predicao:
        acertos += 1
print('acertos:', acertos)
print('taxa de acertos:', acertos / len(teste))