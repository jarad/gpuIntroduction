
my.runif = function(n, ub, ni=1, nd=1, engine="R", seed=1)
{
    engine = pmatch(engine, c("R","C","GPU"))

    switch(engine,
    {
        # R implementation
        u = rep(Inf,n)
        count = rep(0,n)
        set.seed(seed)
        for (i in 1:n) while( (u[i] <- runif(1))>ub ) 
        {
            count[i] = count[i]+1
            a = 0
            b = 1
            for (j in 1:ni) a = a + 1
            for (j in 1:nd) b = b * 1.00001
        }
        return(list(u=u,count=count))
    },
    {
        # C implementation
        set.seed(seed)
        out = .C("cpu_runif_wrap", as.integer(n), as.double(ub), 
                              as.integer(ni), as.integer(nd),
                              u=double(n), count=integer(n))
        return(list(u=out$u,count=out$count))
    },
    {
        # GPU implementation
        out = .C("gpu_runif", as.integer(n), as.double(ub), 
                              as.integer(ni), as.integer(nd),
                              as.double(seed),
                              u=double(n), count=integer(n))
        return(list(u=out$u,count=out$count))
    })
}

