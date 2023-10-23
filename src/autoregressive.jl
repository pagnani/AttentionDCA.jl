function arloss(Q::Array{Float64, 3},
    K::Array{Float64, 3},
    V::Array{Float64, 3}, 
    Z::Matrix{Int},
    weights::Vector{Float64};# ω;
    #reg_version = :CONST, 
    λ::Float64 = 0.001)

    N = size(Z,1)
    q = maximum(Z)
    
    @tullio sf[i, j, h] := Q[h,d,i]*K[h,d,j]
    sf = softmax_notinplace(sf,dims=2) 
    
    @tullio J[i,j,a,b] := sf[i,j,h]*V[h,a,b]*(i>j)
   
    @tullio mat_ene[a,r,m] := J[r,j,a,Z[j,m]]
    lge = logsumexp(mat_ene,dims=1)[1,:,:]

    @tullio pl = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pl = -1*pl


    #if reg_version == :CONST
        reg = λ*(sum(abs2, J))
    #elseif reg_version == :DISTR
    #    reg = λ*ω'*sum(abs2,J,dims=(2,3,4))[:]
    #else
    #    error("Unexcepted value for reg_version")
    #end
    
    
    pl = pl + reg

    return pl
end 

function arloss(m::NamedTuple{(:Q, :K, :V), Tuple{Array{Float64, 3}, Array{Float64, 3}, Array{Float64, 3}}}, 
    Z::Matrix{Int}, 
    weights::Vector{Float64};
    #ω; 
    #reg_version = :CONST,
    λ::Float64 = 0.001)
    
    N = size(Z,1)
    q = maximum(Z)
    
    @tullio sf[i, j, h] := m.Q[h,d,i]*m.K[h,d,j]
    sf = softmax_notinplace(sf,dims=2) 
    
    @tullio J[i,j,a,b] := sf[i,j,h]*m.V[h,a,b]*(i>j)
   
    @tullio mat_ene[a,r,m] := J[r,j,a,Z[j,m]]
    lge = logsumexp(mat_ene,dims=1)[1,:,:]

    @tullio pl = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pl = -1*pl
    

    #if reg_version == :CONST
        reg = λ*(sum(abs2, J))
    #elseif reg_version == :DISTR
    #    reg = λ*ω'*sum(abs2,J,dims=(2,3,4))[:]
    #else
    #    error("Unexcepted value for reg_version")
    #end    
    
    pl = pl + reg

    return pl
end 

# function artrainer(D,η,batch_size,n_epoch; 
#     H = 32,
#     d = 23, 
#     init = rand,
#     λ=0.001, 
#     structfile = "../ArDCAData/data/PF00014/PF00014_struct.dat",
#     savefile::Union{String, Nothing} = nothing)
    
#     N,_ = size(D[1])
#     q = maximum(D[1])

#     m = (Q = init(H,d,N), K = init(H,d,N), V = init(H,q,q))
#     t = setup(Adam(η), m)

#     savefile !== nothing && (file = open(savefile,"a"))
    
#     for i in 1:n_epoch
#         loader = DataLoader(D, batchsize = batch_size, shuffle = true)
#         for (z,w) in loader
#             _w = w/sum(w)
#             g = gradient(x->arloss(x.Q, x.K, x.V, z, _w, λ=λ),m)[1];
#             update!(t,m,g)
#         end

#         #s = score(m.Q,m.K,m.V);
#         #PPV = compute_PPV(s,structfile)
#         l = round(arloss(m.Q, m.K, m.V, D[1], D[2], λ=λ),digits=5) 
#         #p = round((PPV[N]),digits=3)
#         #println("Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
#         println("Epoch $i loss = $l")
#         savefile !== nothing && println(file, "Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
#     end

#     savefile !== nothing && close(file)
#     p0 = computep0(D)
    
#     arnet = arnet_builder(m,p0)

#     return arnet, m
# end

# function artrainer(m,D,η,batch_size,n_epoch;
#     λ = 0.001, 
#     structfile = "../ArDCAData/data/PF00014/PF00014_struct.dat",
#     savefile::Union{String, Nothing} = nothing)

#     N,_ = size(D[1])
#     t = setup(Adam(η), m)
    
#     savefile !== nothing && (file = open(savefile,"a"))

#     for i in 1:n_epoch
#         loader = DataLoader(D, batchsize = batch_size, shuffle = true)
    
#         for (z,w) in loader
#             _w = w/sum(w)
#             g = gradient(x->arloss(x.Q, x.K, x.V, z, _w, λ=λ),m)[1];
#             update!(t,m,g)
#         end
    
#         #s = score(m.Q,m.K,m.V);
#         #PPV = compute_PPV(s,structfile)
#         l = round(arloss(m.Q, m.K, m.V, D[1], D[2], λ=λ),digits=5) 
#         #p = round((PPV[N]),digits=3)
#         #println("Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
#         println("Epoch $i loss = $l")
#         savefile !== nothing && println(file, "Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
#     end

#     savefile !== nothing && close(file)
#     p0 = computep0(D)
#     net = arnet_builder(m, p0)

#     return net, m
# end


# function artrainer2(D,η,batch_size,n_epoch; 
#     H = 32,
#     d = 23, 
#     init = rand,
#     λ=0.001, 
#     structfile = "../ArDCAData/data/PF00014/PF00014_struct.dat",
#     savefile::Union{String, Nothing} = nothing)
    
#     N,_ = size(D[1])
#     q = maximum(D[1])

#     m = (Q = init(H,d,N), K = init(H,d,N), V = init(H,q,q))
#     t = setup(Adam(η), m)

#     savefile !== nothing && (file = open(savefile,"a"))
    
#     for i in 1:n_epoch
#         loader = DataLoader(D, batchsize = batch_size, shuffle = true)
#         for (z,w) in loader
#             _w = w/sum(w)
#             g = gradient(x->arloss2(x.Q, x.K, x.V, z, _w, λ=λ),m)[1];
#             update!(t,m,g)
#         end

#         #s = score(m.Q,m.K,m.V);
#         #PPV = compute_PPV(s,structfile)
#         l = round(arloss2(m.Q, m.K, m.V, D[1], D[2], λ=λ),digits=5) 
#         #p = round((PPV[N]),digits=3)
#         #println("Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
#         println("Epoch $i loss = $l")
#         savefile !== nothing && println(file, "Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
#     end

#     savefile !== nothing && close(file)
#     p0 = computep0(D)
#     net = arnet_builder2(m, p0)

#     return net, m
# end

# function artrainer2(m,D,η,batch_size,n_epoch;
#     λ = 0.001, 
#     structfile = "../ArDCAData/data/PF00014/PF00014_struct.dat",
#     savefile::Union{String, Nothing} = nothing)

#     N,_ = size(D[1])
#     t = setup(Adam(η), m)
    
#     savefile !== nothing && (file = open(savefile,"a"))

#     for i in 1:n_epoch
#         loader = DataLoader(D, batchsize = batch_size, shuffle = true)
    
#         for (z,w) in loader
#             _w = w/sum(w)
#             g = gradient(x->arloss2(x.Q, x.K, x.V, z, _w, λ=λ),m)[1];
#             update!(t,m,g)
#         end
    
#         #s = score(m.Q,m.K,m.V);
#         #PPV = compute_PPV(s,structfile)
#         l = round(arloss2(m.Q, m.K, m.V, D[1], D[2], λ=λ),digits=5) 
#         #p = round((PPV[N]),digits=3)
#         #println("Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
#         println("Epoch $i loss = $l")
#         savefile !== nothing && println(file, "Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
#     end

#     savefile !== nothing && close(file)
#     p0 = computep0(D)
#     net = arnet_builder2(m, p0)

#     return net, m
# end

# function arnet_builder(m, p0; q = 21)
#     _,_,N = size(m.Q)
#     mask = tril(ones(N,N),-1) 
    
#     @tullio W[h, i, j] := m.Q[h,d,i]*m.K[h,d,j]
#     W = softmax(W,dims=3) 
#     @tullio J[i,j,a,b] := W[h,i,j]*m.V[h,a,b]*mask[i,j]
#     J_reshaped = AttentionBasedPlmDCA.reshapetensor(J,N,q)
#     H = [zeros(q) for _ in 1:N-1]
#     net = ArNet(collect(1:N), p0, J_reshaped,H)
#     return net
# end
# function arnet_builder2(m, p0; q = 21)
#     _,_,N = size(m.Q)

#     @tullio W[h, i, j] := m.Q[h,d,i]*m.K[h,d,j] - 1.0e8*(j>=i)
#     W = softmax(W,dims=3)
#     @tullio J[i,j,a,b] := W[h,i,j]*m.V[h,a,b]*(i!=1)
#     J_reshaped = AttentionBasedPlmDCA.reshapetensor(J,N,q)
#     H = [zeros(q) for _ in 1:N-1]
#     net = ArNet(collect(1:N), p0, J_reshaped,H)
#     return net
# end


function artrainer(D::Tuple{Matrix{Int}, Vector{Float64}}, n_epochs::Int, idxperm::Vector{Int}; 
    init_m = Nothing,  
    #reg_version = :CONST,
    η = 0.005, 
    batch_size = 1000, 
    H = 32,
    d = 23, 
    init = rand,
    λ=0.001, 
    savefile::Union{String, Nothing} = nothing)
    


    N,M = size(D[1])
    q = maximum(D[1])

    arvar = ArVar(N,M,q,λ,0.0,D[1],D[2],idxperm)


    m = if init_m !== Nothing
        init_m
    else
        (Q = init(H,d,N), K = init(H,d,N), V = init(H,q,q))
    end
    
    omega = [(i-1)*q*q for i in 1:N]
    omega = omega ./ sum(omega)

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
        
        println("Epoch $i loss = $l")
        
        savefile !== nothing && println(file, "Epoch $i loss = $l")
    end

    savefile !== nothing && close(file)
    p0 = computep0(D)
    
    @tullio W[h, i, j] := m.Q[h,d,i]*m.K[h,d,j]
    W = softmax(W,dims=3) 
    @tullio J[i,j,a,b] := W[h,i,j]*m.V[h,a,b]*(j<i)
    J_reshaped = AttentionBasedPlmDCA.reshapetensor(J,N,q)
    F = [zeros(q) for _ in 1:N-1]
    
    arnet = ArNet(idxperm, p0, J_reshaped,F)

    return arnet, arvar, m
end


function artrainer(filename::String, n_epochs::Int;
    permorder::Union{Symbol, Vector{Int}} = :NATURAL, 
    theta::Union{Symbol,Real}=:auto,
    max_gap_fraction::Real=0.9,
    remove_dups::Bool=true,
    kwds...)
    
    time = @elapsed Weights, Z, N, M, q = ReadFasta(filename, max_gap_fraction, theta, remove_dups)
    
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

    #ArDCA.permuterow!(Z,idxperm) 
    data = (Z,Weights)
    println("preprocessing took $time seconds")
    artrainer(data, n_epochs, idxperm; kwds...)
end







function arloss2(Q::Array{Float64, 3},
    K::Array{Float64, 3},
    V::Array{Float64, 3}, 
    Z::Matrix{Int}, 
    weights::Vector{Float64};
    verbose::Bool = false,  
    λ::Float64 = 0.001)
    
    @tullio sf[i, j, h] := Q[h,d,i]*K[h,d,j] - 1.0e4*(j>=i)
    sf = softmax_notinplace(sf,dims=2)
    
    #sf[1,:,:] .= 0.0
    @tullio J[i,j,a,b] := sf[i,j,h]*V[h,a,b]*(i!=1)
   
    @tullio mat_ene[a,r,m] := J[r,j,a,Z[j,m]]
    lge = AttentionBasedPlmDCA.logsumexp(mat_ene,dims=1)[1,:,:]

    @tullio pl = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pl = -1*pl
    reg = λ*(sum(abs2, J))
    
    verbose && println(pl," + ",reg," = ",pl+reg)
    pl = pl + reg
    
    return pl
end


arloss2(m::NamedTuple{(:Q, :K, :V), Tuple{Array{Float64, 3}, Array{Float64, 3}, Array{Float64, 3}}}, Z::Matrix{Int}, weights::Vector{Float64}; 
    λ::Float64 = 0.001) = arloss2(m..., Z, weights, λ = λ)


function artrainer2(D::Tuple{Matrix{Int}, Vector{Float64}}, n_epochs::Int, idxperm::Vector{Int}; 
    init_m = Nothing,  
    H = 32,
    d = 23,
    batch_size = 1000,
    η = 0.005, 
    init = rand,
    λ=0.001, 
    savefile::Union{String, Nothing} = nothing)
    

    N,M = size(D[1])
    q = maximum(D[1])

    arvar = ArVar(N,M,q,λ,0.0,D[1],D[2],idxperm)

    m = if init_m !== Nothing
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
            g = gradient(x->arloss2(x.Q, x.K, x.V, z, _w, λ=λ),m)[1];
            update!(t,m,g)
        end

        l = round(arloss2(m.Q, m.K, m.V, D[1], D[2], λ=λ),digits=5) 
        println("Epoch $i loss = $l")
        savefile !== nothing && println(file, "Epoch $i loss = $l")
    end

    savefile !== nothing && close(file)
    p0 = computep0(D)
    
    @tullio W[h, i, j] := m.Q[h,d,i]*m.K[h,d,j] - 1.0e8*(j>=i)
    W = softmax(W,dims=3)
    @tullio J[i,j,a,b] := W[h,i,j]*m.V[h,a,b]*(i!=1)
    J_reshaped = AttentionBasedPlmDCA.reshapetensor(J,N,q)
    F = [zeros(q) for _ in 1:N-1]
    
    arnet = ArNet(idxperm, p0, J_reshaped, F)

    return arnet, arvar, m
end


function artrainer2(filename::String, n_epochs::Int;
    permorder::Union{Symbol, Vector{Int}} = :NATURAL,
    theta::Union{Symbol,Real}=:auto,
    max_gap_fraction::Real=0.9,
    remove_dups::Bool=true,
    kwds...)
    
    time = @elapsed Weights, Z, N, M, q = ReadFasta(filename, max_gap_fraction, theta, remove_dups)
    
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
    println("preprocessing took $time seconds")
    
    artrainer2(data, n_epochs, idxperm; kwds...)
end




function stat_artrainer(filename::String, n_sim::Int;
    n_epochs = 100,
    kwds...)
    Z,W = AttentionBasedPlmDCA.quickread(filename)
    s = []
    for _ in 1:n_sim
        m = artrainer((Z,W), n_epochs, [1:size(Z,1);]; kwds...)
        push!(s, epistatic_score(m[1], m[2], 1))
    end
    s = vcat(s...)
    return unique(x->x[1:2],sort(s, by = x -> x[3], rev = true))
end