function loss(Q::Array{T1, 3},
    K::Array{T1, 3},
    V::Array{T1, 3}, 
    Z::Matrix{T2},
    weights::Vector{T3};
    λ::Float64 = 0.001) where {T1<:Real, T2<: Integer, T3<:Real}


    #compute softmax function of Q*K'
    @tullio sf[i, j, h] := Q[h,d,i]*K[h,d,j]
    sf = softmax_notinplace(sf,dims=2) 
    
    #compute J tensor
    @tullio J[i,j,a,b] := sf[i,j,h]*V[h,a,b]*(j!=i)

    #compute the energy of the sequences and the partition function
    @tullio mat_ene[a,r,m] := J[r,j,a,Z[j,m]]
    lge = logsumexp(mat_ene,dims=1)[1,:,:]

    @tullio pl = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pl = -1*pl
    
    #compute the loss
    reg = λ*(sum(abs2, J))
    
    pl = pl + reg

    return pl
end 

loss(m::NamedTuple{(:Q, :K, :V)}, Z::Matrix{T2}, weights::Vector{T1}; kwds...) where {T1<: Real, T2 <: Integer} = loss(m..., Z, weights; kwds...) 

function trainer(D::Tuple{Matrix{T1}, Vector{T2}},n_epochs::Int; 
    H::Int = 32,
    d::Int = 23,
    batch_size::Int = 1000,
    η::Float64 = 0.005,
    λ::Float64 = 0.001,
    init_m = nothing, 
    init_fun = rand, 
    structfile::String = "precompilation_data/PF00014_struct.dat",
    verbose = true,
    savefile::Union{String, Nothing} = nothing) where {T1<:Integer, T2<:Real}
    
    N,_ = size(D[1])
    q = maximum(D[1])

    m = if init_m !== nothing
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
        verbose && println("Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
        savefile !== nothing && println(file, "Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
    end

    savefile !== nothing && close(file)
    return m
end

function trainer(filename::String, n_epochs::Int;
    theta::Union{Symbol,Real}=:auto,
    max_gap_fraction::Real=0.9,
    remove_dups::Bool=true,
    verbose = true,
    kwds...)
    
    time = @elapsed Weights, Z, _, _, _ = ReadFasta(filename, max_gap_fraction, theta, remove_dups, verbose = verbose)
    
    data = (Z,Weights)
    verbose && println("preprocessing took $time seconds")
    trainer(data, n_epochs; verbose = verbose, kwds...)
end



function stat_trainer(filename::String, n_sim::Int;
    n_epochs::Int = 100,
    verbose = true,
    kwds...)
    Z,W = AttentionDCA.quickread(filename)
    s = []
    for _ in 1:n_sim
        m = trainer((Z,W), n_epochs; verbose = verbose, kwds...)
        push!(s,score(m...))
    end
    s = vcat(s...)
    return unique(x->x[1:2],sort(s, by = x -> x[3], rev = true))
end
 
