

rm(list=ls())
#Seasonal ARIMA Models
install.packages("jpeg")
require(FinTS)
library(tseries)
library(forecast)
library(Ecdat) #It may require install package jpeg
data(IncomeUK) 
consumption = IncomeUK[,2] 


#Problem 1 Describe the behavior of consumption. What types of differencing, seasonal, nonseasonal, or both, would you recommend? Do you recommend fitting a seasonal ARIMA model to the data with or without a log transformation? Consider also using ACF plots to help answer these questions.
plot(consumption)
acf(consumption)
adf.test(consumption , k=4)

plot(diff(consumption)) #Differencing the consumption time series by order 1
acf(diff(consumption))  #ACF plot of the same
acf(diff(consumption),lag.max=20) # ACF plot if the same with 20 lags
adf.test(consumption)

logConsumption=log(consumption) #consumption is a time series
plot(logConsumption) #plotting the logConsumption time series 
acf(logConsumption,lag.max = 20) # ACF of the logConsumption of the time series with lag = 20
pacf(logConsumption,lag.max =20)

#Problem 2 Regardless of your answers to Problem 1, find an ARIMA model that provides a good fit to log(consumption). What order model did you select? (Give the orders of the nonseasonal and seasonal components.)
plot(diff(logConsumption))
acf(diff(logConsumption),lag.max = 20)
pacf(diff(logConsumption))

?arima()
model = arima(logConsumption, order = c(0,1,0),seasonal=list(order=c(0,1,1),period=4)) #autoarima model using AIC
model
#Problem 3 Check the ACF of the residuals from the model you selected in Problem 2. Do you see any residual autocorrelation?
acf(model$residuals) #ACF plot of residuals
Box.test(model$residuals, lag=20, type = "Ljung") # Ljung box test for the residuals 


#Problem 4 Apply auto.arima to log(consumption) using BIC. What model is selected?
model2= auto.arima(logConsumption, ic = 'bic') # auto arima model using BIC
summary(model2)

#Problem 5 Forecast log(consumption) for the next eight quarters using the models you found in Problems 2 and 4. Plot the two sets of forecasts in side-by- side plots with the same limits on the x- and y-axes. Describe any differences between the two sets of forecasts. Using the backshift operator, write the models you found in problems 2 and 4.
fit_AA_AIC = auto.arima(logConsumption,ic="aic") #Preparing the model
fit_AA_BIC = auto.arima(logConsumption,ic="bic") #Preparing the model

forecast_aa_aic = forecast(fit_AA_AIC,h=8) #Forecasting
forecast_aa_bic = forecast(fit_AA_BIC,h=8) #Forecasting

par(mfrow=c(1,2))
plot(forecast_aa_aic$mean,xlim=c(1985.5,1987.25),ylim=c(10.86,11))#AIC plot
plot(forecast_aa_bic$mean,xlim=c(1985.5,1987.25),ylim=c(10.86,11))#BIC Plot


#Note: To predict an arima object (an object returned by the arima function), use the predict function. To learn how the predict function works on an arima object, use ?predict.Arima. To forecast an object returned by auto.arima, use the forecast function in the forecast package. For example, the following code will forecast eight quarters ahead using the object returned by auto.arima and then plot the forecasts.

# fitAutoArima = auto.arima(logConsumption,ic="bic") 
# foreAutoArima = forecast(fitAutoArima,h=8) 
# plot(foreAutoArima$mean,xlim=c(1985.5,1987.25),ylim=c(10.86,11))

#Problem 6 Include the variable include log(Income) as an exogenous variable to forecast log(consumption) using auto.arima.  According to the AIC, is this model better than the previous models? (Hint: use xreg to include exogenous variables in arima and auto.arima)
IncomeUK
IncomeUK[,1]
logincome=log(IncomeUK[,1])#Converting to log for xreg
logincome
?auto.arima()
m3=auto.arima(logConsumption,ic="aic",stepwise = FALSE, approximation = FALSE, xreg = logincome) #fitting the model
m3

predict=forecast(m3,h=20,xreg=logincome) #predicitng
predict$mean
plot(predict$mean)
autoplot(predict$mean) #plotting the model
#Your submission should include a file with your R code and another UNCOMPRESSED file with your report where you answer the above questions including the appropriate graphs. Please do not submit a compressed file.