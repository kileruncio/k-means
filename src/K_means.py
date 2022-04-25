import numpy as np
import pandas as pd

class K_Means:
    def __init__(self, k, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        
    def calculate(self, dataset):
        self.centroids = {}
        
        for i in range(self.k):
            self.centroids[i] = dataset[i]
            
        for i in range(self.max_iter):
            self.classifications = {}
            
            for a in range(self.k):
                self.classifications[a] = []
                
            for data in dataset:
                distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(data)
            
            k_centroids = dict(self.centroids)
            
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
                
            done = True
            
            for centroid in self.centroids:
                original = k_centroids[centroid]
                current = self.centroids[centroid]
                if np.sum((current-original)/original*100.0) > self.tol:
                    print(np.sum((current-original)/original*100.0))
                    done = False
                    
            if done:
                break
            
    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
    
    def non_numeric_to_numeric(self, file):
        columns = file.columns.values
        
        for column in columns:
            values = {}
            
            def convert_to_int(val):
                return values[val]
            
            if file[column].dtype != np.int64 and file[column].dtype != np.float64:
                col_content = file[column].values.tolist()
                unique_elements = set(col_content)
                
                x = 0
                
                for unique in unique_elements:
                    if unique not in values:
                        values[unique] = x
                        x += 1
                
                file[column] = list(map(convert_to_int, file[column]))
                
        return file