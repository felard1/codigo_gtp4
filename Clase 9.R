library(ISLR)
fix(Hitters)


data=na.omit(Hitters)
dim(data)

data=data[,c(-14,-15,-20)]



########################################################################
#Para que sea mas claro, partimos la muestra en dos: train y test
#Usamos el MSE con la muestra test

set.seed(49)
n1=sample(seq(1:dim(data)[1]),dim(data)[1]-100)
train=data[n1,]
test=data[-n1,]


########################################################################
#1. Recordando el modelo con el mejor subconjunto de variables
library(leaps)

subs=regsubsets(Salary~.,train,nvmax=14)
sum_subs=summary(subs)
plot(sum_subs$cp)
which.min(sum_subs$cp)
sum_subs

lm1=lm(Salary~AtBat+Hits+Runs+Walks+CHmRun+CRuns+PutOuts,data=train)
summary(lm1)
pred=predict(lm1,test)
mse=mean((test$Salary-pred)^2)
mse



########################################################################
#2. Con PCAR
library(pls)

lm2=pcr(Salary~.,data=train,scale=T,validation="CV",ncomp=16)
summary(lm2)
predpp=predict(lm2,test)
msepp=mean((test$Salary-predpp)^2)
msepp



########################################################################
#4. Con PLSR
library(pls)


lm3=plsr(Salary~.,data=train,scale=T,validation="CV")
summary(lm3)
predpl=predict(lm3,test)
msepl=mean((test$Salary-predpl)^2)
msepl



########################################################################
#5. Comparando los tres metodos

mse
msepp
msepl


########################################################################
#6. Ahora con penalizacion caudratica (Ridge)

library(glmnet)

X=model.matrix(Salary~.,train)[,-1]
Xtest=model.matrix(Salary~.,test)[,-1]

y=train$Salary
ytest=test$Salary

cvmod=cv.glmnet(X,y,alpha=0)
cvmod$lambda.min
plot(cvmod)
#Grafico de betas
mod_pen2_plot=glmnet(X,y,alpha=0)
plot(mod_pen2_plot,xvar=c("lambda"))
#plot(mod_pen2_plot,xvar=c("dev"))



mod_pen2=glmnet(X,y,alpha=0,lambda=cvmod$lambda.min)
coef(mod_pen2)

predp2=predict(mod_pen2,Xtest)
msep2=mean((ytest-predp2)^2)
msep2



#Comparacion de resultados con modelos lineales
mse
msepp
msepl
msep2



########################################################################
#7. Ahora con penalizacion en norma 1 (LASSO)

library(glmnet)

mod1=glmnet(X,y,alpha=1)
plot(mod1)


cvmod1=cv.glmnet(X,y,alpha=1)
cvmod1$lambda.min
plot(cvmod1)

mod_pen1=glmnet(X,y,alpha=1,lambda=cvmod1$lambda.min)
coef(mod_pen1)

predp1=predict(mod_pen1,Xtest)
msep1=mean((ytest-predp1)^2)
msep1



#Comparacion de resultados con modelos lineales
mse
msepp
msepl
msep2
msep1

#------------------------------------------
#############################################################
#Lasso High Dimensional
#############################################################
#Regresion

#parametros y dimensiones
n=2000
p=1500
x=matrix(rnorm(n*p),n,p)
index=sample.int(p,p)
beta=c(runif(floor(p/4),-8,8),rep(0,p-floor(p/4)))
beta=as.matrix(beta[index])

y=x%*%beta+rnorm(n,sd=1)
plot(y,x[,3])

#Estimacion
#OLS
fitlm=lm(y~x)
diff=beta-fitlm$coefficients[1:p+1]
mean(abs(diff))

#LASSO
library(glmnet)

lass=glmnet(x,y)
plot(lass,"dev")
plot(lass,"lambda")
lass=glmnet(x,y,lambda=.2)
blass=lass$beta


#variable selection
lass_index=ifelse(blass>0,1,0)
beta_index=ifelse(beta>0,1,0)
sum(abs(lass_index-beta_index))

#Cross-validation
cvlass=cv.glmnet(x,y,type.measure="mse")
plot(cvlass)




#############################################################
#Lasso High Dimensional
#############################################################
#Real Data: VAriable Selection
#General lienar regression

library(devtools)
#install_github('ramhiser/datamicroarray')

data("alon",package="datamicroarray")
data=alon

dim(data$x)
head(data$x)

data$y

cvlass=cv.glmnet(data$x,data$y,family="binomial",type.measure="auc",nfolds=5)
plot(cvlass)

glasso=glmnet(data$x,data$y,family="binomial")
plot(glasso)
glasso=glmnet(data$x,data$y,family="binomial",lambda=exp(-2.5))
blasso=glasso$beta

#---------------------------------------------

#############################################################
#Regresion
library(glmnet)
library(mvtnorm)
library(corrplot)
set.seed(54)

#parametros y dimensiones
n=500
p=50
rho=0.8
sigma=matrix(rep(rho,10*10),ncol=10) ; diag(sigma)=rep(1,10)
x1=rmvnorm(n, mean = rep(0, nrow(sigma)), sigma = sigma)
x2=rmvnorm(n, mean = rep(0, nrow(sigma)), sigma = sigma)
x3=rmvnorm(n, mean = rep(0, nrow(sigma)), sigma = sigma)
x4=rmvnorm(n, mean = rep(0, nrow(sigma)), sigma = sigma)
x5=rmvnorm(n, mean = rep(0, nrow(sigma)), sigma = sigma)
x=cbind(x1,x2,x3,x4,x5)
corx=cor(x); corrplot(corx)
#index=sample.int(p,p)
index=seq(1,50)
beta=c(runif(30,-1,1),rep(0,p-30))
beta=as.matrix(beta[index])

y=x%*%beta+rnorm(n,sd=2)
plot(y,x[,3])


#LASSO
library(glmnet)

lass=glmnet(x,y)
plot(lass,"dev")
plot(lass,"lambda",
     col=c(rep(1,10),rep(2,10),rep(3,10),rep(4,10),rep(5,10)))

#Cross-validation
cvlass=cv.glmnet(x,y,type.measure="mse")
plot(cvlass)

lass=glmnet(x,y,lambda=exp(-1))
blass=lass$beta

#variable selection
lass_index=ifelse(blass>0,1,0)
beta_index=ifelse(beta>0,1,0)
sum(abs(lass_index-beta_index))


#ELASTIC NET
enet=glmnet(x,y,alpha=0.1)
plot(enet,"dev",
     col=c(rep(1,10),rep(2,10),rep(3,10),rep(4,10),rep(5,10)))

enet=glmnet(x,y,lambda=exp(1),alpha=0.2)
benet=enet$beta


####################################################################
#Group LASSO
library(grplasso)

ind=as.factor(c(rep(1,10),rep(2,10),rep(3,10),rep(4,10),rep(5,10)))
xy=data.frame(y,x)

lambda=seq(10,350,by=30)

grlass=grplasso(y~ .,data=xy, model = LinReg(), lambda = lambda,contrast=ind, 
                center = TRUE, standardize = TRUE)

grlass$coefficients
#---------------------------------------------------

library(corrplot)
library(glmnet)
library(grplasso)


bank=read.csv("bankruptcy_data.csv",head=T)
bank=bank[,-1]
bank=bank[,-(2:6)]
bank=na.omit(bank)
corx=cor(bank[,-1]); corrplot(corx)

#Elastic net analysis
y=as.factor(bank$FRACASO)
x=as.matrix(bank[,-1])
ent=glmnet(x,y,family="binomial")
plot(ent)
cvlass=cv.glmnet(x,y,family="binomial",type.measure="auc",nfolds=5,alpha=.8)
plot(cvlass)
ent=glmnet(x,y,family="binomial",lambda=exp(-6),alpha=.8)
ent$beta


#Group lasso analysis

groups=c(rep(1,4),rep(2,4),rep(3,3),rep(4,4),rep(5,3),rep(6,11),rep(7,5))
groups=as.factor(groups)

fit.bank <- grplasso(FRACASO ~ ., data = bank, model = LogReg(), 
                     lambda = seq(1,100,by=10),contrasts = groups, center = TRUE, standardize = TRUE)

summary(fit.bank)
fit.bank$coefficients
plot(fit.bank)


#---------------------------------------
############################################################
#1. Funtion Definition:
#You can try to modify the denominator inside the cosine to 
#change the complexity of the function

f=function(x){
  y=2+x^(.92)*cos(x/.3)/x^.05
  return(y)
}
plot(f,0,5)



############################################################
#2. Data Simulation:
#You can try to change the sample size or the variance
N=300
sigma=1
x=runif(N,0,5); x=sort(x)
y=rep(0,times=N)
for(i in 1:N){
  y[i]=f(x[i])+rnorm(1,0,sigma)
}
plot(x,y)
points(x,f(x),type="l",col=2,lwd=2)

#Note that the x-data points are sorted.  That was made for 
#plotting, not because it is necessary for estimation



############################################################
#3. Estimation by k-neighbors:
#You can try to change the sample size or the variance

#Estimation Function
# The input parameters are: k (number of neighbors), test 
#(x points for tetisn), and the traninng sample (x and y)

kn=function(k,test,x,y){
  z=test;ll=length(z)
  nk=rep(0,times=ll)
  for(j in 1:ll){
    veci=which(abs(z[j]-x) %in% sort(abs(z[j]-x))[1:k])
    nk[j]=sum(y[veci])/k
  }
  return(nk)
}

# the output of the function is the predicted hat(f) at
# each testing point
# You could use this new code for the homework 1



############################################################
#4. Effect of Cross-Validation:

#4.1. Splitting the sample in 50-50 (training-testing)

#The data will be rearrange randomly (because the x s are
#sorted).

fold=2
ss=seq(1:N)
ss=sample(ss,N,replace=F)
ss1=ss[1:(N/fold)]
ss2=ss[(N/fold+1):N]


y_train=y[ss2]
x_train=x[ss2]
y_test=y[ss1]
x_test=x[ss1]


#MSE test dataset
k=seq(1:100)
kk=length(k)

mse_test=rep(0,times=kk)

for(i in 1:kk){
  out=kn(k[i],x_test,x_train,y_train)
  mse_test[i]=mean((y_test-out)^2)
}

plot(k,mse_test,type="l")
points(k,mse_test,type="l",col=2)


#Now you can change the splitting:
#Repeat running again ss=sample(ss,N,replace=F)
#Plot an look the differences



#4.2. For 10-fold cross-validation:

#In each iteration, you take the 10th part as
#the training sample.  

#For one iteration:

fold=10

ss1=ss[1:((fold-1)*N/fold)]
ss2=ss[((fold-1)*N/fold+1):N]

#Create the MSE curves
#In homework, for 10-fold cross-validation,
#you have to repeat for each fold and
#averaging the 10 MSE curves

