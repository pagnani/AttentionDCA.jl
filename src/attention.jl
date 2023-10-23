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

loss(m::NamedTuple{(:Q, :K, :V)}, Z::Matrix{Int}, weights::Vector{Float64}; kwds...) = loss(m..., Z, weights; kwds...) 

function trainer(D,n_epochs; 
    H = 32,
    d = 23,
    batch_size = 1000,
    η = 0.005,
    λ = 0.001,
    init_m = Nothing, 
    init_fun = rand, 
    structfile = "../ArDCAData/data/PF00014/PF00014_struct.dat",
    savefile::Union{String, Nothing} = nothing)
    
    N,_ = size(D[1])
    q = maximum(D[1])

    m = if init_m !== Nothing
        init_m
    else
        (Q = init_fun(H,d,N), K = init_fun(H,d,N), V = init_fun(H,q,q))
    end
    t = setup(Adam(η), m)

    savefile !== nothing && (file = open(savefile,"a"))
    
    for i in 1:n_epochs
        loader = DataLoader(D, batchsize = batch_size, shuffle = true)
        for (z,w) in loader
            _w = w/sum(w)
            g = gradient(x->loss(x.Q, x.K, x.V, z, _w, λ = λ),m)[1];
            update!(t,m,g)
        end

        s = score(m.Q,m.K,m.V);
        PPV = compute_PPV(s,structfile)
        l = round(loss(m.Q, m.K, m.V, D[1], D[2], λ = λ),digits=5) 
        p = round((PPV[N]),digits=3)
        println("Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
        savefile !== nothing && println(file, "Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
    end

    savefile !== nothing && close(file)
    return m
end

function trainer(filename::String, n_epochs::Int;
    theta::Union{Symbol,Real}=:auto,
    max_gap_fraction::Real=0.9,
    remove_dups::Bool=true,
    kwds...)
    
    time = @elapsed Weights, Z, N, M, q = ReadFasta(filename, max_gap_fraction, theta, remove_dups)
    
    data = (Z,Weights)
    println("preprocessing took $time seconds")
    
    trainer(data, n_epochs; kwds...)
end


function stat_trainer(filename::String, n_sim::Int;
    Z,W = AttentionBasedPlmDCA.quickread(filename)
    n_epochs = 100,
    kwds...)
    s = []
    for _ in 1:n_sim
        m = trainer((Z,W), n_epochs; kwds...)
        push!(s,score(m...))
    end
    s = vcat(s...)
    return unique(x->x[1:2],sort(s, by = x -> x[3], rev = true))
end