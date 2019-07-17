#import the dependencies
from random import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
#Generate randomized dataset in the range of 1 to 1000
dataset = [x + random() for x in range(1, 1000)]
# fitting the model, notice that by not specifying the exofgenous parameter (exog=None) the model becomes SARIMA
sarima = SARIMAX(dataset, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
sarima_fit = sarima.fit(disp=False)
# make  the prediction
y = sarima_fit.predict(len(dataset), len(dataset))
print(y)

#try to fiddle with the parameters of the SARIMA model