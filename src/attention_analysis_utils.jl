function attention_matrix_sym(Q,K)
    H,_,N = size(Q)
    @tullio sf[h,i,j] := Q[h,d,i]*K[h,d,j]
    sf = AttentionBasedPlmDCA.softmax(sf,dims=3)
    @tullio sf[h,i,j] *= (i!=j)
    
    W = zeros(H,N,N)
    
    for h in 1:H
        W[h,:,:] = (sf[h,:,:] + sf[h,:,:]')/2
    end
    
    return W
        
end

function attention_matrix_asym(Q,K)
    H,_,N = size(Q)
    @tullio sf[h,i,j] := Q[h,d,i]*K[h,d,j]
    sf = AttentionBasedPlmDCA.softmax(sf,dims=3)
    @tullio sf[h,i,j] *= (i!=j)
    
    return sf
        
end

attention_matrix_asym(Q,K,V) = attention_matrix_asym(Q,K)
attention_matrix_sym(Q,K,V) = attention_matrix_sym(Q,K)

function freezedVtrainer(D,V,n_epochs; 
    η = 0.005,
    batch_size = 1000,
    d = 23, 
    init = rand,
    λ = 0.001,
    structfile = "../ArDCAData/data/PF00014/PF00014_struct.dat",
    savefile::Union{String, Nothing} = nothing)
    
    N,_ = size(D[1])
    H,q,q = size(V)
    
    V = V[1:H, :, :] #nel caso la V esterna abbia più teste di quelle che usiamo per la famiglia corrente

    m = (Q = init(H,d,N), K = init(H,d,N))
    t = setup(Adam(η), m)

    savefile !== nothing && (file = open(savefile,"a"))
    
    for i in 1:n_epochs
        loader = DataLoader(D, batchsize = batch_size, shuffle = true)
        for (z,w) in loader
            _w = w/sum(w)
            g = gradient(x->loss(x.Q, x.K, V, z, _w, λ = λ),m)[1];
            update!(t,m,g)
        end

        s = score(m.Q,m.K,V);
        PPV = compute_PPV(s,structfile)
        l = round(loss(m.Q, m.K, V, D[1], D[2], λ = λ),digits=5) 
        p = round((PPV[N]),digits=3)
        println("Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
        savefile !== nothing && println(file, "Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
    end

    savefile !== nothing && close(file)
    return m
end


function freezedVtrainer(filename::String, V::Array{Float64,3}, n_epochs::Int;
    theta::Union{Symbol,Real}=:auto,
    max_gap_fraction::Real=0.9,
    remove_dups::Bool=true,
    kwds...)
    
    time = @elapsed Weights, Z, N, M, q = ReadFasta(filename, max_gap_fraction, theta, remove_dups)
    
    data = (Z,Weights)
    println("preprocessing took $time seconds")
    
    freezedVtrainer(data,V,n_epochs; kwds...)
end

function quickread(fastafile; moreinfo=false)
    
    Weights, Z, N, M, _ = ReadFasta(fastafile, 0.9, :auto, true, verbose = false);
    moreinfo && return Weights, Z, N, M
    return Z, Weights
    
end




###########################################
########### AR VERSION ####################
###########################################


function ar_freezedVtrainer(D::Tuple{Matrix{Int}, Vector{Float64}}, V, n_epochs::Int, idxperm::Vector{Int}; 
    init_m = Nothing,  
    #reg_version = :CONST,
    η::Float64 = 0.005,
    batch_size::Int = 1000, 
    d = 23, 
    init = rand,
    λ=0.001, 
    savefile::Union{String, Nothing} = nothing)
    


    N,M = size(D[1])
    H,q,_ = szie(V)

    arvar = ArVar(N,M,q,λ,0.0,D[1],D[2],idxperm)


    m = if init_m !== Nothing
        init_m
    else
        (Q = init(H,d,N), K = init(H,d,N))
    end
    
    omega = [(i-1)*q*q for i in 1:N]
    omega = omega ./ sum(omega)

    t = setup(Adam(η), m)

    savefile !== nothing && (file = open(savefile,"a"))
    
    for i in 1:n_epochs
        loader = DataLoader(D, batchsize = batch_size, shuffle = true)
        for (z,w) in loader
            _w = w/sum(w)
            g = gradient(x->arloss(x.Q, x.K, V, z, _w, λ=λ),m)[1];
            update!(t,m,g)
        end

        l = round(arloss(m.Q, m.K, V, D[1], D[2],  λ=λ),digits=5) 
        
        println("Epoch $i loss = $l")
        
        savefile !== nothing && println(file, "Epoch $i loss = $l")
    end

    savefile !== nothing && close(file)
    p0 = computep0(D)
    
    @tullio W[h, i, j] := m.Q[h,d,i]*m.K[h,d,j]
    W = softmax(W,dims=3) 
    @tullio J[i,j,a,b] := W[h,i,j]*V[h,a,b]*(j<i)
    J_reshaped = AttentionBasedPlmDCA.reshapetensor(J,N,q)
    F = [zeros(q) for _ in 1:N-1]
    
    arnet = ArNet(idxperm, p0, J_reshaped,F)

    return arnet, arvar, m
end


function ar_freezedVtrainer(filename::String, V::Array{Float64,3}, n_epochs::Int, permorder::Union{Symbol, Vector{Int}};
    theta::Union{Symbol,Real}=:auto,
    max_gap_fraction::Real=0.9,
    remove_dups::Bool=true,
    kwds...)
    
    time = @elapsed Weights, Z, N, M, q = ReadFasta(filename, max_gap_fraction, theta, remove_dups)
    
    idxperm = if typeof(permorder) == Symbol
        S = entropy(Z,Weights)
        if permorder === :ENTROPIC
            sortperm(S)
        elseif permorder === :REV_ENTROPIC
            sortperm(S,rev=true)
        elseif permorder === :RANDOM
            randperm(N)
        elseif permorder === :NATURAL
            collect(1:N)
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
    ar_freezedVtrainer(data, V, n_epochs, idxperm; kwds...)
end


function my_epistatic_score(Q,K,V, arvar;
    permorder = :NATURAL, 
    min_separation::Int=6)

    H,d,N = size(Q)
    @extract arvar : Z Weights = W M N q 

    idxperm = if typeof(permorder) == Symbol
        S = entropy(Z,Weights)
        if permorder === :ENTROPIC
            sortperm(S)
        elseif permorder === :REV_ENTROPIC
            sortperm(S,rev=true)
        elseif permorder === :RANDOM
            randperm(N)
        elseif permorder === :NATURAL
            collect(1:N)
        else
            error("the end of the world has come")
        end
    elseif typeof(permorder) <: Vector
        (length(permorder) != N) && error("length permorder ≠ $N")
        isperm(permorder) && (permorder)
    else
        error("permorder can only be a Symbol or a Vector")
    end

    all_ep_score = []
    
    @tullio W[h, i, j] := Q[h,d,i]*K[h,d,j]
    W = softmax(W,dims=3) 
    p0 = computep0((Z,Weights))


    for h in 1:H 
        @tullio J[i,j,a,b] := W[$h,i,j]*V[$h,a,b]*(j<i)
        J_reshaped = AttentionBasedPlmDCA.reshapetensor(J,N,q)
        F = [zeros(q) for _ in 1:N-1]
        arnet = ArNet(idxperm, p0, J_reshaped,F)
        push!(all_ep_score, epistatic_score(arnet, arvar, rand(1:N), min_separation = min_separation))
    end

    return all_ep_score

end