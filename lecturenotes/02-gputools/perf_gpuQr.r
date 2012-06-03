library(multcomp)
library(gputools)

cpu_function = qr
gpu_function = gpuQr

r = 2 # number of replicates for each size of matrix x
c = 100000 # columns of matrix x
n = 10 # number of sizes of matrix x to try
m = 1000000 # each size matrix has n * m rows
params = rep(1:n,each=r)

xs = (1:n) * m * c
ys = list()

xlab = "Number of Matrix Entries"
title = "qr() vs gpuQr()"
plot.name = "performance_gpuQr"

cols = list(cpu = "blue", gpu = "green", outlier.gpu = "black")


iter.time = function(param, type = "cpu"){
    x <- matrix(rnorm(m * param), ncol = c)

    if(type == "cpu"){
      ptm <- proc.time()
      cpu_function(x)
      ptm <- proc.time() - ptm
    } else{
      ptm <- proc.time()
      gpu_function(x)
      ptm <- proc.time() - ptm
    }

    return(list(user = ptm[1], syst = ptm[2], elap = ptm[3]))
}


loop.time = function(params, type = "cpu"){

  user = c()
  syst = c()
  elap = c()

  for(i in params){
    times = iter.time(param = i, type = type)    

    user = c(user, times$user)
    syst = c(syst, times$syst)
    elap = c(elap, times$elap)
  }

  return(list(user = user, syst = syst, elap = elap))
}

cpu.times = loop.time(params, type = "cpu")
outlier.gpu.times = loop.time(params[1], type = "gpu") 
gpu.times = loop.time(params, type = "gpu")

times = list(cpu = cpu.times,
             outlier.gpu = outlier.gpu.times,
             gpu = gpu.times)

for(dev in c("cpu","gpu")){
  for(time in c("user","syst","elap")){
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


for(time in c("user", "syst", "elap")){
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
       ylab = paste(c(time, "runtime", collapse = " ")),
       main = paste(c(time, "runtime:", title, collapse = " ")))  

  for(dev in c("cpu", "gpu")){
    points(xs[1], ys[[time]]$outlier.gpu, col=cols$outlier.gpu)
    points(xs, ys[[time]][[dev]]$Estimate, col = cols[[dev]])
    lines(xs, ys[[time]][[dev]]$lwr, col = cols[[dev]], lty=1)
    lines(xs, ys[[time]][[dev]]$upr, col = cols[[dev]], lty=1)
  }

  legend("topleft",
         legend = c("mean cpu runtime", 
                    "mean gpu runtime", 
                    "first gpu run (overhead, discarded)",
                    "95% CI bound"),
         col = c(cols$cpu,
                 cols$gpu,
                 "black", "black"),
         pch = c("o", "o", "o", "."),
         lty = c(0,0,0,1))

  dev.off()
}