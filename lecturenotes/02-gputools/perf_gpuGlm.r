# Name: performance.r
# Author: Will Landau (landau@iastate.edu)
# Created: June 2012
#
# This program calculates the runtime of 
# a user-specified function, gpu_function,
# and compares it to that of some cpu
# analog, cpu_function.  
#
# The script creates three plots, each 
# comparing the runtimes of gpu_function 
# to those of cpu_function based on either
# user time, system time, or total scheduled time.


library(multcomp)
library(gputools)


#############
## GLOBALS ##
#############

chooseGpu(1)

# functions to compare
cpu_function = glm
gpu_function = gpuGlm

# global runtime parameters
r = 2 # number of replicates for each size of matrix x
c = 100 # columns of matrix x
n = 10 # number of sizes of matrix x to try
m = 10000 # each size matrix has n * m rows
xs = (1:n) * m * c
ys = list()
xlab = "Number of Matrix Entries"
title = "glm() vs gpuGlm()"
plot.name = "performance_gpuGlm"
cols = list(cpu = "blue", gpu = "green", outlier.gpu = "black")

# parameter vector: each entry defines the computational
# load of gpu_function (or cpu_function) for a single run


####################
## MAIN FUNCTIONS ##
####################

# iter.time() is a wrapper for one iteration of either 
# gpu_function or cpu_function. The purpose of the wrapper
# is to create the input data, pass the appropriate
# parameter (entry param of params), and return the run time.
iter.time = function(param, type = "cpu"){
  x <- matrix(rnorm(m * param), ncol = c)
  y <- rnorm(dim(x)[1])
  
  if(type == "cpu"){
    ptm <- proc.time()
    cpu_function(y~x)
    ptm <- proc.time() - ptm
  } else{
    ptm <- proc.time()
    gpu_function(y~x)
    ptm <- proc.time() - ptm
  }
  
  return(list(user = ptm[1], syst = ptm[2], total = ptm[3]))
}

# loop.time executes iter.time (i.e., calculates the run time 
# of either gpu_function or cpu_function) for each entry in 
# params (one run per entry in params, each entry defining 
# the magnitude of the computational load on gpu_function or 
# cpu_function).
loop.time = function(params, type = "cpu"){
  
  user = c()
  syst = c()
  total = c()
  
  for(i in params){
    times = iter.time(param = i, type = type)    
    
    user = c(user, times$user)
    syst = c(syst, times$syst)
    total = c(total, times$total)
  }
  
  return(list(user = user, syst = syst, total = total))
}


##################
## MAIN ROUTINE ##
##################

# Main routine: actually run gpu_function and cpu_function
# for various data loads and return the run times. Note:
# outlier.gpu.times measures the computational overhead 
# associated with using the gpu for the first time in this
# R script
cpu.times = loop.time(params, type = "cpu")
outlier.gpu.times = loop.time(params[1], type = "gpu") 
gpu.times = loop.time(params, type = "gpu")


#########################################
## FORMAT RUNTIME DATA OF MAIN ROUTINE ##
#########################################

# organize runtime data into a convenient list
times = list(cpu = cpu.times,
             outlier.gpu = outlier.gpu.times,
             gpu = gpu.times)


# Format runtime data for plotting: calculate family-wise 
# confidence regions for each run time type
# (time = "user", "syst", or "total") and
# each device (dev = "cpu", "gpu", or 
# "outlier.gpu"). The data, ready for plotting, 
# are available for each device in ys[[time]][[dev]].
for(dev in c("cpu","gpu")){
  for(time in c("user","syst","total")){
    fit = aov(times[[dev]][[time]] ~ as.factor(params) - 1)
    glht.fit = glht(fit)
    
    if(!all(glht.fit$coef == 0)){
      famint = confint(glht.fit)
    } else{
      zeroes = rep(0,length(glht.fit$coef))
      famint = list(confint = list(Estimate = zeroes,
                                   lwr = zeroes,
                                   upr = zeroes))
    }
    
    ys[[time]][[dev]] = data.frame(famint$confint)
    ys[[time]]$outlier.gpu = times$outlier.gpu[[time]]
  }
}


#######################
## PLOT RUNTIME DATA ##
#######################

# For each kind of run time, make a plot comparing
# the run times of gpu_function to the run times
# of cpu_function.
for(time in c("user", "syst", "total")){
  filename = paste(c(plot.name,"_",time,".pdf"), collapse = "")
  pdf(filename)
  
  xbounds = c(min(xs), max(xs))
  ybounds = c(min(unlist(ys[[time]])),
              1.3 * max(unlist(ys[[time]])))
  
  plot(xbounds,
       ybounds,
       pch= ".",
       col="white",
       xlab = xlab,
       ylab = paste(c(time, "scheduled runtime", collapse = " ")),
       main = paste(c(time, "scheduled runtime:", title, collapse = " ")))  
  
  for(dev in c("cpu", "gpu")){
    points(xs[1], ys[[time]]$outlier.gpu, col=cols$outlier.gpu)
    points(xs, ys[[time]][[dev]]$Estimate, col = cols[[dev]])
    lines(xs, ys[[time]][[dev]]$lwr, col = cols[[dev]], lty=1)
    lines(xs, ys[[time]][[dev]]$upr, col = cols[[dev]], lty=1)
  }
  
  legend("topleft",
         legend = c("mean cpu runtime", 
                    "mean gpu runtime", 
                    "first gpu run (overhead, discarded from conf. region calculations)",
                    "95% CI bound"),
         col = c(cols$cpu,
                 cols$gpu,
                 "black", "black"),
         pch = c("o", "o", "o", "."),
         lty = c(0,0,0,1))
  
  dev.off()
}