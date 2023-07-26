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
    H = 32,
    d = 23, 
    init = rand,
    λ = 0.001,
    structfile = "../ArDCAData/data/PF00014/PF00014_struct.dat",
    savefile::Union{String, Nothing} = nothing)
    
    N,_ = size(D[1])
    #q = maximum(D[1])
    
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