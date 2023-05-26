function loss(Q::Array{Float64, 3},
    K::Array{Float64, 3},
    V::Array{Float64, 3}, 
    Z::Matrix{Int}, 
    weights::Vector{Float64}; 
    λ::Float64 = 0.001)

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

function loss(m::NamedTuple{(:Q, :K, :V), Tuple{Array{Float64, 3}, Array{Float64, 3}, Array{Float64, 3}}}, 
    Z::Matrix{Int}, 
    weights::Vector{Float64}; 
    λ::Float64 = 0.001)
    
    @tullio sf[i, j, h] := m.Q[h,d,i]*m.K[h,d,j] #O(NNHd)
    sf = AttentionBasedPlmDCA.softmax_notinplace(sf,dims=2) 
    
    @tullio J[i,j,a,b] := sf[i,j,h]*m.V[h,a,b]*(j!=i) #O(NNqqH)
   
    @tullio mat_ene[a,r,m] := J[r,j,a,Z[j,m]] #O(qNNM)
    lge = logsumexp(mat_ene,dims=1)[1,:,:]

    @tullio pl = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m]) #O(NM)
    pl = -1*pl
    reg = λ*(sum(abs2, J))
    
    pl = pl + reg

    return pl
end 

function trainer(D,η,batch_size,n_epoch; 
    H = 32,
    d = 23, 
    init = rand, 
    structfile = "../ArDCAData/data/PF00014/PF00014_struct.dat",
    savefile::Union{String, Nothing} = nothing)
    
    N,_ = size(D[1])
    q = maximum(D[1])

    m = (Q = init(H,d,N), K = init(H,d,N), V = init(H,q,q))
    t = setup(Adam(η), m)

    savefile !== nothing && (file = open(savefile,"a"))
    
    for i in 1:n_epoch
        loader = DataLoader(D, batchsize = batch_size, shuffle = true)
        for (z,w) in loader
            _w = w/sum(w)
            g = gradient(x->loss(x.Q, x.K, x.V, z, _w),m)[1];
            update!(t,m,g)
        end

        s = score(m.Q,m.K,m.V);
        PPV = compute_PPV(s,structfile)
        l = round(loss(m.Q, m.K, m.V, D[1], D[2]),digits=5) 
        p = round((PPV[N]),digits=3)
        println("Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
        savefile !== nothing && println(file, "Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
    end

    savefile !== nothing && close(file)
    return m
end

function trainer(m,D,η,batch_size,n_epoch; 
    structfile = "../ArDCAData/data/PF00014/PF00014_struct.dat",
    savefile::Union{String, Nothing} = nothing)

    N,_ = size(D[1])
    t = setup(Adam(η), m)
    
    savefile !== nothing && (file = open(savefile,"a"))

    for i in 1:n_epoch
        loader = DataLoader(D, batchsize = batch_size, shuffle = true)
    
        for (z,w) in loader
            _w = w/sum(w)
            g = gradient(x->loss(x.Q, x.K, x.V, z, _w),m)[1];
            update!(t,m,g)
        end
    
        s = score(m.Q,m.K,m.V);
        PPV = compute_PPV(s,structfile)
        l = round(loss(m.Q, m.K, m.V, D[1], D[2]),digits=5) 
        p = round((PPV[N]),digits=3)
        println("Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
        savefile !== nothing && println(file, "Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
    end

    savefile !== nothing && close(file)

    return m
end

function readfile!(file)    
    f = open(file, "r")

    pslike = []
    precision = []
    firsterror = []

    for line in eachline(file)
        a = split(line)
        push!(pslike, parse(Float64,a[5]))
        push!(precision, parse(Float64,a[8]))
        push!(firsterror, parse(Int,a[end]))
    end

    close(f)

    return pslike, precision, firsterror
end







function arloss(Q::Array{Float64, 3},
    K::Array{Float64, 3},
    V::Array{Float64, 3}, 
    Z::Matrix{Int}, 
    weights::Vector{Float64}; 
    λ::Float64 = 0.001)

    N = size(Z,1)
    mask = tril(ones(N,N),-1) 
    
    @tullio sf[i, j, h] := Q[h,d,i]*K[h,d,j]
    sf = softmax_notinplace(sf,dims=2) 
    
    @tullio J[i,j,a,b] := sf[i,j,h]*V[h,a,b]*mask[i,j]
   
    @tullio mat_ene[a,r,m] := J[r,j,a,Z[j,m]]
    lge = logsumexp(mat_ene,dims=1)[1,:,:]

    @tullio pl = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pl = -1*pl
    reg = λ*(sum(abs2, J))
    
    pl = pl + reg

    return pl
end 

function arloss(m::NamedTuple{(:Q, :K, :V), Tuple{Array{Float64, 3}, Array{Float64, 3}, Array{Float64, 3}}}, 
    Z::Matrix{Int}, 
    weights::Vector{Float64}; 
    λ::Float64 = 0.001)
    
    N = size(Z,1)
    mask = tril(ones(N,N),-1) 
    
    @tullio sf[i, j, h] := m.Q[h,d,i]*m.K[h,d,j]
    sf = softmax_notinplace(sf,dims=2) 
    
    @tullio J[i,j,a,b] := sf[i,j,h]*m.V[h,a,b]*mask[i,j]
   
    @tullio mat_ene[a,r,m] := J[r,j,a,Z[j,m]]
    lge = logsumexp(mat_ene,dims=1)[1,:,:]

    @tullio pl = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pl = -1*pl
    reg = λ*(sum(abs2, J))
    
    pl = pl + reg

    return pl
end 

function artrainer(D,η,batch_size,n_epoch; 
    H = 32,
    d = 23, 
    init = rand, 
    structfile = "../ArDCAData/data/PF00014/PF00014_struct.dat",
    savefile::Union{String, Nothing} = nothing)
    
    N,_ = size(D[1])
    q = maximum(D[1])

    m = (Q = init(H,d,N), K = init(H,d,N), V = init(H,q,q))
    t = setup(Adam(η), m)

    savefile !== nothing && (file = open(savefile,"a"))
    
    for i in 1:n_epoch
        loader = DataLoader(D, batchsize = batch_size, shuffle = true)
        for (z,w) in loader
            _w = w/sum(w)
            g = gradient(x->arloss(x.Q, x.K, x.V, z, _w),m)[1];
            update!(t,m,g)
        end

        #s = score(m.Q,m.K,m.V);
        #PPV = compute_PPV(s,structfile)
        l = round(arloss(m.Q, m.K, m.V, D[1], D[2]),digits=5) 
        #p = round((PPV[N]),digits=3)
        #println("Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
        println("Epoch $i loss = $l")
        savefile !== nothing && println(file, "Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
    end

    savefile !== nothing && close(file)
    return m
end

function artrainer(m,D,η,batch_size,n_epoch; 
    structfile = "../ArDCAData/data/PF00014/PF00014_struct.dat",
    savefile::Union{String, Nothing} = nothing)

    N,_ = size(D[1])
    t = setup(Adam(η), m)
    
    savefile !== nothing && (file = open(savefile,"a"))

    for i in 1:n_epoch
        loader = DataLoader(D, batchsize = batch_size, shuffle = true)
    
        for (z,w) in loader
            _w = w/sum(w)
            g = gradient(x->arloss(x.Q, x.K, x.V, z, _w),m)[1];
            update!(t,m,g)
        end
    
        #s = score(m.Q,m.K,m.V);
        #PPV = compute_PPV(s,structfile)
        l = round(arloss(m.Q, m.K, m.V, D[1], D[2]),digits=5) 
        #p = round((PPV[N]),digits=3)
        #println("Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
        println("Epoch $i loss = $l")
        savefile !== nothing && println(file, "Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
    end

    savefile !== nothing && close(file)

    return m
end