import pandas as pd
import numpy as np
from K_means import K_Means 


file_name = input('Provide relative or absolute path to the file: ')
k = int(input('Provide k: '))

cols = pd.read_csv(file_name, nrows = 1).columns.tolist()
file = pd.read_csv(file_name, skiprows = 1, names = cols[:-1] + ['name'])
logic = K_Means(k)

dataset = logic.non_numeric_to_numeric(file)
correct_answers = np.array(dataset['name'])
dataset.drop(['name'], 1, inplace = True)
data = np.array(dataset.astype(float))
logic.calculate(data)   
correct = 0

for i in range(len(data)):
    predict = np.array(data[i].astype(float))
    predict = predict.reshape(-1, len(predict))
    result = logic.predict(predict)
    print(f'{np.array(data[i].astype(float))} {result}')
    
    if result == correct_answers[i]:
        correct += 1
print(f'Accuracy: {(correct/len(data))*100 if (correct/len(data))*100 > 50 else (1.0-(correct/len(data)))*100}%')
