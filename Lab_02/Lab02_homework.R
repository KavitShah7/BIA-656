### Kavit Shah ###
# Homework - 2

rm(list=ls())
setwd("C:\\Kavit\\Stevens Institute of Technology\\SEM 3\\BIA 656 Advanced Data Analytics and Machine Learning") #<== set my working directory
library(fGarch)
da=read.table("d-spy-0111.txt",header=T)
srtn=log(da$rtn+1) ### Log returns
srtn
acf(srtn)
pacf(srtn)

t.test(srtn)# t-test to check the means of the 2 samples
m1=garchFit(~garch(1,1),data=srtn)  # lots of output
m1=garchFit(~garch(1,1),data=srtn,trace=F) # no output printed.
summary(m1) # Obtain results and model checking statistics
## Should understand the meaning of model checking
sresi=m1@residuals/m1@sigma.t  ## For model checking
acf(sresi)
Box.test(sresi,lag=12,type="Ljung")
Box.test(sresi^2,lag=12,type="Ljung")

#Evaluate Arch Effect:
require(FinTS)
#Complete dataset:
ArchTest(srtn)  #Evaluate Arch effect: Computes the Lagrange multiplier test for conditional heteroscedasticity of Engle (1982) 
#Standardized residuals:
ArchTest(sresi)


# plot(m1)  #<=== Many choices are available, choice 3 and 13 are useful.
# predict(m1,6) # prediction
# 
# # use Student-t innovations
# m2=garchFit(~garch(1,1),data=srtn,trace=F,cond.dist="std")
# summary(m2)
# 
# # use skewed Student-t innovations
# m3=garchFit(~garch(1,1),data=srtn,trace=F,cond.dist="sstd")
# summary(m3) ### Should understand how to check the skewness: p-value < 0.05 reject H0: symmetric distribution --> skewness





##### ARMA-GARCH with QQ-standard residual plot ##### 

# Use ARMA(1,1)+GARCH(1,1) model with normal innovations (in case needs an ARMA model for mean eq.)
m4=garchFit(~arma(1,1)+garch(2,1),data=srtn,trace=F)
summary(m4)
sresidual=m4@residuals/m4@sigma.t       # For model checking
acf(sresidual)                          # Acf plots for the residuals
Box.test(sresidual,lag=12,type="Ljung") # Ljung box-test for residual significance testing
plot(m4)                                #QQplot for the standard residuals 

##### ARMA-GARCH with STUDENT-T Innovations ##### 


# c)
m4=garchFit(~arma(1,1)+garch(2,1),data=srtn,trace=F,cond.dist = "std")
summary(m4)
sresidual=m4@residuals/m4@sigma.t
Box.test(sresidual,lag=12,type="Ljung")
acf(sresidual) 

# ### The following two models are identical.
# m5=garchFit(~garch(1,1),data=srtn,trace=F,leverage=T)
# summary(m5)


#### APARCH with 1-5 step predictions and voaltility  
# spec1=garchFit(~arma(1,1)+aparch(1,1),data=srtn,trace=F,delta=2,include.delta=F,cond.dist="std")
# summary(m6)
require(rugarch)
spec1=ugarchspec(variance.model=list(model="apARCH"),distribution.model="std",
                 mean.model=list(armaOrder=c(1,1)))
m6=ugarchfit(spec=spec1,data=srtn)
m6  ### see output

### prediction, 1-step to 5-step ahead
p1 <- ugarchforecast(m6,n.ahead=5)
p1
sigma(p1) ### volatility prediction
fitted(p1) ### mean prediction








#Question 2 

rm(list=ls())
setwd("C:\\Kavit\\Stevens Institute of Technology\\SEM 3\\BIA 656 Advanced Data Analytics and Machine Learning") #<== set my working directory
library(fGarch)
require(FinTS)
da=read.table("m-ko-6111.txt",header=T)
da
srtn1=log(da$ko+1) ### Log returns
srtn1
acf(srtn1)
pacf(srtn1)
ArchTest(srtn1)



m1=garchFit(~garch(1,1),data=srtn1)
standardresi=m1@residuals/m1@sigma.t  ## For model checking
acf(standardresi)
Box.test(standardresi,lag=12,type="Ljung")





require(rugarch)
spec2=ugarchspec(variance.model=list(model="sGARCH"),distribution.model="std")
                 #mean.model=list(armaOrder=c(0,0)))
m1=ugarchfit(spec=spec2,data=srtn1)
m1  ### see output
plot(m1)

### prediction, 1-step to 5-step ahead
predict1 <- ugarchforecast(m1,n.ahead=5)
predict1

sigma(predict1) ### volatility prediction
fitted(predict1) ### 






######## Question - 3 ##########
rm(list=ls())
setwd("C:\\Kavit\\Stevens Institute of Technology\\SEM 3\\BIA 656 Advanced Data Analytics and Machine Learning") #<== set my working directory
library(fGarch)
require(FinTS)
require(rugarch)
da=read.table("m-ko-6111.txt",header=T)
da
srtn1=log(da$ko+1) ### Log returns
srtn2=srtn1*100 # converting to percentage log returns 

spec = ugarchspec(
  variance.model = list(model = "fGARCH", garchOrder = c(1,1), 
                        submodel = "TGARCH"), 
  mean.model = list(armaOrder = c(1,1), include.mean = TRUE))

model=ugarchfit(spec = spec, data=srtn2)
model





































# m1@fit$ics[1]
# m2@fit$ics[1]
# m3@fit$ics[1]  #Best: use skewed Student-t innovations
# m4@fit$ics[1]
# m5@fit$ics[1]
# m6@fit$ics[1]

######################################################################################################
#### Alternative package: rugarch
####
### Using the package "rugarch"
require(rugarch)
#### The following four specifications are helpful. You can copy them when needed.
#### You can modify them if needed once you are familiar with the package.
####
### Specify an GARCH(1,1) model
spec1=ugarchspec(variance.model=list(model="sGARCH",garchOrder=c(1,1)),
                 mean.model=list(armaOrder=c(0,0)))
### Specify an IGARCH(1,1) model
spec2=ugarchspec(variance.model=list(model="iGARCH"),mean.model=list(armaOrder=c(0,0)))
### Specify an eGARCH(1,1) model
spec3=ugarchspec(variance.model=list(model="eGARCH"),mean.model=list(armaOrder=c(0,0)))
### Specify a GJR-GARCH(1,1) model (Threshold GARCH)
spec4=ugarchspec(variance.model=list(model="gjrGARCH"),mean.model=list(armaOrder=c(0,0)))
### Specify a GARCH-M(1,1) model
variance.model=list(model="fGARCH",garchOrder=c(1,1), submodel="GARCH")
spec4=ugarchspec(variance.model=list(model="gjrGARCH"),mean.model=list(armaOrder=c(0,0)))
mean.model=list(armaOrder=c(1,0),include.mean=TRUE,archm=T,archpow=2)
summary(spec4)
spec5=ugarchspec(variance.model=variance.model,mean.model=mean.model, distribution.model="norm")
### Specify a SGARCH(1,1) model with standardized Student-t distribution
spec6=ugarchspec(variance.model=list(model="sGARCH",garchOrder=c(1,1)),
                 mean.model=list(armaOrder=c(0,0)),distribution.model="std")

### Estimation SGARCH
m7=ugarchfit(spec=spec1,data=srtn)
m7  ### see output
plot(m7) ### There are 12 choices
### prediction, 1-step to 5-step ahead
p1 <- ugarchforecast(m7,n.ahead=5)
p1
sigma(p1) ### volatility prediction
fitted(p1) ### mean prediction
### SGARCH(1,1) model with standardized Student-t distribution
m8 <- ugarchfit(data=srtn,spec=spec6)
m8


### With external.regressors
require(quantmod)
getSymbols("^VIX",from="2007-01-03",to="2016-04-28")
vix <- as.numeric(VIX[,6])/100
getSymbols("AAPL",from="2007-01-03",to="2016-04-28")
rtn <- diff(log(as.numeric(AAPL[,6])))
### use vix index the day before
x1 <- as.matrix(vix[-length(vix)])
spec6 <- ugarchspec(variance.model=list(model="sGARCH",garchOrder=c(1,1),external.regressors=x1),
                    mean.model=list(armaOrder=c(0,0)))
m9 <- ugarchfit(data=rtn,spec=spec6)
m9
### Use the vix of the same day
x2 <- as.matrix(vix[-1])
spec7 <- ugarchspec(variance.model=list(model="sGARCH",garchOrder=c(1,1),external.regressors=x2),
                    mean.model=list(armaOrder=c(0,0)))
m10 <- ugarchfit(data=rtn,spec=spec7)
m10

########## Stochastic volatility models

source("svfit.R") ### Stochastic volatility models.
### The script requires "fGarch" and "mvtnorm" packages
# fits univariate stochastic volatility models without the leverage effect.

m11=svfit(srtn,200,500)
names(m11)
m11
require(stochvol)
sp <- scan(file="sp500.txt")
sp <- sp-mean(sp) ### The program assumes zero mean.
sv1 <- svsample(sp) #simulates from the joint posterior distribution of the SV parameters mu, phi, sigma (and potentially nu), 
#along with the latent log-volatilities h_0,...,h_n and returns the MCMC draws.
names(sv1)
apply(sv1$para,2,mean)  ## posterior mean of parameters
apply(sv1$para,2,var) ## posterion variance of parameters
ht <- apply(sv1$latent,2,median) #latent:	Markov Chain Montecarlo object containing the latent instantaneous log-volatility draws from the posterior distribution.
v1 <- exp(ht/2) ### volatility
ts.plot(v1)
