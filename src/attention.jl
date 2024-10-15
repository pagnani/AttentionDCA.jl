"""
Function to compute the loss of the attention model given (Q,K,V), the MSA and the weight vector of a protein family.\n
If the MSA contains M sequence of length L, encoded with integers from 1 to q, and the attention model has H heads and inner dimension d, then:\n 
    Q and K are HxdxN matrices\n
    V is a Hxqxq matrix\n
    Z is the LxM MSA matrix\n
    W is the M-dimensional weight vector\n
    λ is the regularization parameter\n
"""
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


"""

    trainer(D::Tuple{Matrix{T1}, Vector{T2}},n_epochs::Int,...)

Function to train the attention model given a tuple D = (Z,W) containing the MSA and the weight vector of a protein family, and the number of epochs for the training.
Optional arguments are: \n
    H: number of heads
    d: inner dimension
    η: learning parameter
    λ: regularization parameter
    init_m: initialization for the (Q,K,V) parameters, default nothing 
    init_fun: initialization function for the (Q,K,V) parameters, default rand 
    structfile: file containing the structure of the protein family used for printing the Positive Predicted Value of the model during learning, default nothing 
    savefile: file where to save the log, default nothing

The `structfile` is a file containing a list of (i, j, d_ij) where d_ij is the distance in Angstrom between the residues i and j.\n
It returns a structure out::OutAttStd containing a NamedTuple with the trained model out.m = (Q,K,V).
# Examples
```
julia> out = trainer((Z,W),100,H=32,d=23,η=0.005,λ=0.001,verbose=true,savefile="log.txt");
julia> out.m
```
"""
function trainer(D::Tuple{Matrix{T1}, Vector{T2}},n_epochs::Int; 
    H::Int = 32,
    d::Int = 23,
    batch_size::Int = 1000,
    η::Float64 = 0.005,
    λ::Float64 = 0.001,
    init_m = nothing, 
    init_fun = rand, 
    structfile::Union{Nothing, String} = nothing,
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
    
    function show_info(i,structfile)
        if structfile !== nothing
            PPV = compute_PPV(score(m.Q,m.K,m.V),structfile)
            l = round(loss(m.Q, m.K, m.V, D[1], D[2], λ = λ),digits=5) 
            p = round((PPV[N]),digits=3)
            verbose && println("Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
            savefile !== nothing && println(file, "Epoch 0 loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
        else
            l = round(loss(m.Q, m.K, m.V, D[1], D[2], λ = λ),digits=5) 
            verbose && println("Epoch $i loss = $l")
            savefile !== nothing && println("Epoch 0 loss = $l")
        end
    end


    for i in 1:n_epochs
        loader = DataLoader(D, batchsize = batch_size, shuffle = true)
        for (z,w) in loader
            _w = w/sum(w)
            g = gradient(x->loss(x.Q, x.K, x.V, z, _w, λ = λ),m)[1];
            update!(t,m,g)
        end
        show_info(i, structfile)
    end

    savefile !== nothing && close(file)
    return AttentionDCA.AttOutStd(m, nothing)
end


"""

    trainer(filename::String,n_epochs::Int,...)

Function to train the attention model starting from a fasta file containing the MSA of a protein family.
# Examples
```
julia> out = trainer("file.fasta",100,H=32,d=23,η=0.005,λ=0.001,verbose=true,savefile="log.txt");
julia> out.m
```
"""
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


"""
        stat_trainer(filename::String,n_sim::Int,...)
Function to trainer the model multiple times and return a contact score given by maxiumum through each single shot score for each contact pair.\n
It outputs a structure out::AttOutStd containing the Frobenious score of the family: out.score\n
    n_sim: number of simulations
    n_epochs: number of epochs for each simulation

# Examples
```
julia> out = stat_trainer("file.fasta",20,n_epochs=100);
julia> out.score 
```


"""
function stat_trainer(filename::String, n_sim::Int;
    n_epochs::Int = 100,
    verbose = true,
    kwds...)
    Z,W = AttentionDCA.quickread(filename)
    s = []
    for _ in 1:n_sim
        out = trainer((Z,W), n_epochs; verbose = verbose, kwds...)
        push!(s,score(out.m...))
    end
    s = vcat(s...)
    #return unique(x->x[1:2],sort(s, by = x -> x[3], rev = true))
    return AttentionDCA.AttOutStd(nothing, unique(x->x[1:2],sort(s, by = x -> x[3], rev = true)))
end
 
