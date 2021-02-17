#Name: Kavit Shah
#CWID: 10452991
#HW-01- BIA-656


#Requirementes: packages Ecdat, tseries, forecast, colorspace
install.packages("Edcat")
install.packages("tseries")
install.packages("forecast")
data(Tbrate,package="Ecdat")
library(tseries)
library(forecast)
# r = the 91-day treasury bill rate
# y = the log of real GDP
# pi = the inflation rate

#To show the top of the file:
Tbrate[1:5,]

plot(Tbrate)
adf.test(Tbrate[,1])
acf(Tbrate)

#1. Describe the signs of nonstationarity seen in the time series and ACF plots.
#2. Is it necessary to obtain the first or higher difference of the series?
acf(Tbrate[,1])
acf(diff(Tbrate[,1]))

plot(Tbrate[,1])
plot(diff(Tbrate[,1],lag=1,differences = 1))#to remove the linear trend differences=1
plot(diff(Tbrate[,1],lag=4,differences = 1))#to remove the quarterly trend for the  quarterly plot of lag=4
plot(diff(Tbrate[,1],lag = 4,differences = 2))# to remove the trend and the seasonality of the quarterly plot

pacf(Tbrate[,1])

auto.arima(Tbrate[,1],ic="aic",stepwise = FALSE, approximation = FALSE )

#3. What order of differencing is chosen? 
#4. Does this result agree with your previous conclusions?
#5. What model was chosen by AIC?
#6. Which goodness-of-fit criterion is being used here?
#7. Change the criterion to BIC. Does the best-fitting model then change?

auto.arima(Tbrate[,1],ic="bic",stepwise = FALSE, approximation = FALSE )

#Finally, refit the best-fitting model with the following code, and 
#check for any residual autocorrelation. You will need to replace the three 
#question marks by the appropriate numerical values for the best-fitting model.

fit1 = arima(Tbrate[,1],order=c(0,1,1),seasonal = list(order=c(1,0,1),period=4))
acf(residuals(fit1))
Box.test(residuals(fit1), lag = 10, type="Ljung")

#8. Do you think that there is residual autocorrelation? If so, 
#describe this autocorrelation and suggest a more appropriate model for 
#the T-bill series.


#This example shows how to forecast a time series using R. Run the following
#code to fit a nonseasonal ARIMA model to the quarterly inflation rate. The
#code also uses the predict function to forecast 36 quarters ahead. The standard 
#errors of the forecasts are also returned by predict and can be used
#to create prediction intervals. Note the use of col to specify colors. Replace
#c(?,?,?) by the specification of the ARIMA model that minimizes BIC.


data(Tbrate,package="Ecdat")
dat = as.data.frame(Tbrate)
attach(dat)


adf.test(Tbrate[,3])
acf(Tbrate[,3])
pacf(Tbrate[,3])

library(forecast)

# r = the 91-day Treasury bill rate
# y = the log of real GDP
# pi = the inflation rate
# fit the nonseasonal ARIMA model found by auto.arima

auto.arima(Tbrate[,3],ic="bic",stepwise = FALSE, approximation = FALSE )
fit = arima(pi,order=c(1,1,1))

#if pi is not a time series you may have to run the following command to transform it into a time series:
pi=ts(pi,frequency=4,start=1950)

forecasts = predict(fit,36)
plot(pi,xlim=c(1980,2006),ylim=c(-7,12))
lines(seq(from=1997,by=.25,length=36),
      forecasts$pred,col="red")
lines(seq(from=1997,by=.25,length=36),
      forecasts$pred + 1.96*forecasts$se,
      col="blue")
lines(seq(from=1997,by=.25,length=36),
      forecasts$pred - 1.96*forecasts$se,
      col="blue")

#Include the plot with your work.
#9. Why do the prediction intervals (blue curves) widen as one moves farther
#into the future?
#10. What causes the the predictions (red) and the prediction intervals to wiggle
#initially?

#Your submission should include a file with your R code and another UNCOMPRESSED file with your report where you answer the above questions including the appropriate graphs. Please do not submit a compressed file.
