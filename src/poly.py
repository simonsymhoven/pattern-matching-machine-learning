
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
class PolynomClassification:

    def __init__(self, degree: int) -> None:
        self.degree = degree

    def fit(self,x:np.ndarray,y:np.ndarray):
        self.func =  np.poly1d(np.polyfit(x, y, deg=self.degree))
    
    def predict(self,x:np.ndarray):
        return self.func(x)