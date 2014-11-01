library(MASS)
n <- 1000
mu <- c(0,0)
sigma <- matrix(c(1,0,0,10),2,2)
x <- mvrnorm(n,mu,sigma)
write(t(x),"mvnorm2d.dat",sep="\t",ncolumns=2)

mu <- c(0,0,0)
sigma <- matrix(c(1,0,0,0,10,0,0,0,100),3,3)
x <- mvrnorm(n,mu,sigma)
write(t(x),"mvnorm3d.dat",sep="\t",ncolumns=3)
