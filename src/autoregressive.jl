function arloss(Q::Array{T1, 3},
    K::Array{T1, 3},
    V::Array{T1, 3}, 
    Z::Matrix{T2},
    weights::Vector{T3};
    λ::Float64 = 0.001) where {T1<:Real, T2<: Integer, T3<:Real}
    
    @tullio sf[i, j, h] := Q[h,d,i]*K[h,d,j]
    sf = softmax_notinplace(sf,dims=2) 
    
    @tullio J[i,j,a,b] := sf[i,j,h]*V[h,a,b]*(i>j)
   
    @tullio mat_ene[a,r,m] := J[r,j,a,Z[j,m]]
    lge = logsumexp(mat_ene,dims=1)[1,:,:]

    @tullio pl = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pl = -1*pl

    reg = λ*(sum(abs2, J))

    pl = pl + reg

    return pl
end 

arloss(m::NamedTuple{(:Q, :K, :V)}, Z::Matrix{T1}, weights::Vector{T2}; kwds...) where {T1 <: Integer, T2 <: Real} = arloss(m..., Z, weights; kwds...)

function artrainer(D::Tuple{Matrix{T1}, Vector{T2}}, n_epochs::Int, idxperm::Vector{Int}; 
    init_m = nothing,
    η::Float64 = 0.005, 
    batch_size::Int = 1000, 
    H::Int = 32,
    d::Int = 23, 
    init = rand,
    λ::Float64 = 0.001,
    verbose = true, 
    savefile::Union{String, Nothing} = nothing) where {T1<:Integer, T2<:Real}
    


    N,M = size(D[1])
    q = maximum(D[1])

    arvar = ArVar(N,M,q,λ,0.0,D[1],D[2],1/M,idxperm)


    m = if init_m !== nothing
        init_m
    else
        (Q = init(H,d,N), K = init(H,d,N), V = init(H,q,q))
    end
    
    t = setup(Adam(η), m)

    savefile !== nothing && (file = open(savefile,"a"))
    
    for i in 1:n_epochs
        loader = DataLoader(D, batchsize = batch_size, shuffle = true)
        for (z,w) in loader
            _w = w/sum(w)
            g = gradient(x->arloss(x.Q, x.K, x.V, z, _w, λ=λ),m)[1];
            update!(t,m,g)
        end

        l = round(arloss(m.Q, m.K, m.V, D[1], D[2],  λ=λ),digits=5) 
        
        verbose && println("Epoch $i loss = $l")
        
        savefile !== nothing && println(file, "Epoch $i loss = $l")
    end

    savefile !== nothing && close(file)
    p0 = computep0(D)
    
    @tullio W[h, i, j] := m.Q[h,d,i]*m.K[h,d,j]
    W = softmax(W,dims=3) 
    @tullio J[i,j,a,b] := W[h,i,j]*m.V[h,a,b]*(j<i)
    J_reshaped = AttentionDCA.reshapetensor(J,N,q)
    F = [zeros(q) for _ in 1:N-1]
    
    arnet = ArNet(idxperm, p0, J_reshaped,F)

    return arnet, arvar, m
end

function artrainer(filename::String, n_epochs::Int;
    permorder::Union{Symbol, Vector{Int}} = :NATURAL, 
    theta::Union{Symbol,Real}=:auto,
    max_gap_fraction::Real=0.9,
    remove_dups::Bool=true,
    verbose = true,
    kwds...)
    
    time = @elapsed Weights, Z, N, M, q = ReadFasta(filename, max_gap_fraction, theta, remove_dups, verbose = verbose)
    
    idxperm = if typeof(permorder) == Symbol
        if permorder === :NATURAL
            collect(1:N)
        elseif permorder === :ENTROPIC
            S = entropy(Z,Weights)
            sortperm(S)
        elseif permorder === :REV_ENTROPIC
            S = entropy(Z,Weights)
            sortperm(S,rev=true)
        elseif permorder === :RANDOM
            randperm(N)
        else
            error("the end of the world has come")
        end
    elseif typeof(permorder) <: Vector
        (length(permorder) != N) && error("length permorder ≠ $N")
        isperm(permorder) && (permorder)
    else
        error("permorder can only be a Symbol or a Vector")
    end

    data = (Z,Weights)
    verbose && println("preprocessing took $time seconds")
    artrainer(data, n_epochs, idxperm; verbose = verbose, kwds...)
end

function stat_artrainer(filename::String, n_sim::Int;
    n_epochs = 100,
    verbose = true,
    kwds...)
    Z,W = AttentionDCA.quickread(filename)
    s = []
    for _ in 1:n_sim
        m = artrainer((Z,W), n_epochs, [1:size(Z,1);]; verbose = verbose, kwds...)
        push!(s, epistatic_score(m[1], m[2], 1))
    end
    s = vcat(s...)
    return unique(x->x[1:2],sort(s, by = x -> x[3], rev = true))
end