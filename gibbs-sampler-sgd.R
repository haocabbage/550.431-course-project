rm(list=ls())
n = 20 # number of pixels per line
sweep=100 # number of swweps for gibbs sampler
M = 100 #number of steps for estimating the parameters
z = matrix(0,nrow=n,ncol=n)
u = matrix(0,nrow=n,ncol=n)
lambda=-0.01
beta=0.2
lambda.hat=vector(length=M)
beta.hat = vector(length=M)
gamma.0=0.1
#####################################################
## functions                       ##################
#####################################################
phi = function(x) return(1/(1+exp(-x)))
gibbs = function(lambda,beta){
  z[,]=rbinom(n=n*n,size=1,prob=0.5)
  z=2*z-1
  for (sw in (1:sweep)){
    for (x in (1:n)){
      for (y in (1:n)){
        neb=z[(x-2)%%n+1,y]+z[(x)%%n+1,y]+
        z[x,(y-2)%%n+1]+z[x,(y)%%n+1]
        p = phi(2*lambda+ 2*beta*neb)
        z[x,y]=2*rbinom(n=1,size=1,prob=p)-1
      }
    }
  }
  return(z)
}
pair.sum=function(z){
  s=0
  for (x in (1:(n-1)))
    for (y in (1:(n-1))){
      s=s+z[x,y]*z[x,y+1]
      s=s+z[x,y]*z[x+1,y]
    }
  return(s)
}
image.plot=function(z){
  z=0.5*(z+1)
  image(x=1:n,y=1:n,z,col=gray((0:256)/256))
}  
######################################################
## generate 1 image ##################################
######################################################
z=gibbs(lambda,beta)
image.plot(z)
#####################################################
## estimate the parameters ##########################
#####################################################
lambda.hat[1]=0
beta.hat[1]=0
for (i in (2:M)){
  gamma=gamma.0
  u = gibbs(lambda.hat[i-1],beta.hat[i-1])
  lambda.hat[i]=lambda.hat[i-1]+gamma*sum(z-u)/(n*n)
  beta.hat[i]=beta.hat[i-1]+gamma*(pair.sum(z)-pair.sum(u))/(2*n*n)
}