#import the dependencies
from random import random
from statsmodels.tsa.arima_model import ARIMA
#Generate randomized dataset in the range of 1 to 1000
dataset = [x + random() for x in range(1, 1000)]
# # fitting the model
arima = ARIMA(dataset, order=(1, 1, 1))
arima_fit = arima.fit(disp=False)
# make prediction
y = arima_fit.predict(len(dataset), len(dataset), typ='levels')
print(y)

#try to fiddle with the parameters of the ARIMA model