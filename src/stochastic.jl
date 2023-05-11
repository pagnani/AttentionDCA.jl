function myloss(Q,K,V, Z, weights; λ = 0.001)

    @tullio sf[i, j, h] := Q[h,d,i]*K[h,d,j]
    sf = softmax_notinplace(sf,dims=2) 
    
    @tullio J[i,j,a,b] := sf[i,j,h]*V[h,a,b]*(j!=i)
   
    @tullio mat_ene[a,r,m] := J[r,j,a,Z[j,m]]
    lge = logsumexp(mat_ene,dims=1)[1,:,:]

    @tullio pl = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pl = -1*pl
    reg = λ*(sum(abs2, J))
    
    pl = pl + reg

    return pl
end 

function myloss(m, Z, weights; λ = 0.001)
    
    @tullio sf[i, j, h] := Q[h,d,i]*K[h,d,j]
    sf = AttentionBasedPlmDCA.softmax_notinplace(sf,dims=2) 
    
    @tullio J[i,j,a,b] := sf[i,j,h]*V[h,a,b]*(j!=i)
   
    @tullio mat_ene[a,r,m] := J[r,j,a,Z[j,m]]
    lge = logsumexp(mat_ene,dims=1)[1,:,:]

    @tullio pl = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pl = -1*pl
    reg = λ*(sum(abs2, J))
    
    pl = pl + reg

    return pl
end 

function mytrainer(D,η,batch_size,n_epoch; 
    H = 32,
    d = 23, 
    init = rand, 
    structfile = "../ArDCAData/data/PF00014/PF00014_struct.dat",
    savefile::Union{String, Nothing} = Nothing)
    
    N,_ = size(D[1])
    q = maximum(D[1])

    m = (Q = init(H,d,N), K = init(H,d,N), V = init(H,q,q))
    t = setup(Adam(η), m)

    savefile != Nothing && (file = open(savefile,"a"))
    
    for i in 1:n_epoch
        loader = DataLoader(D, batchsize = batch_size, shuffle = true)
        for (z,w) in loader
            _w = w/sum(w)
            g = gradient(x->myloss(x.Q, x.K, x.V, z, _w),m)[1];
            update!(t,m,g)
        end

        s = score(m.Q,m.K,m.V);
        PPV = compute_PPV(s,structfile)
        l = round(myloss(m.Q, m.K, m.V, D[1], D[2]),digits=5) 
        p = round((PPV[N]),digits=3)
        println("Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
        println(file, "Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
    end

    savefile != Nothing && close(file)
    return m
end

function mytrainer(m,D,η,batch_size,n_epoch; 
    structfile = "../ArDCAData/data/PF00014/PF00014_struct.dat",
    savefile::Union{String, Nothing} = Nothing)

    N,_ = size(D[1])
    t = setup(Adam(η), m)
    
    savefile != Nothing && (file = open(savefile,"a"))

    for i in 1:n_epoch
        loader = DataLoader(D, batchsize = batch_size, shuffle = true)
    
        for (z,w) in loader
            _w = w/sum(w)
            g = gradient(x->myloss(x.Q, x.K, x.V, z, _w),m)[1];
            update!(t,m,g)
        end
    
        s = score(m.Q,m.K,m.V);
        PPV = compute_PPV(s,structfile)
        l = round(myloss(m.Q, m.K, m.V, D[1], D[2]),digits=5) 
        p = round((PPV[N]),digits=3)
        println("Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
        println(file, "Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
    end

    savefile != Nothing && close(file)

    return m
end
