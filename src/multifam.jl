
"""
    multi_trainer(D::Vector{Tuple{Matrix{Int}, Vector{Float64}}}, n_epochs::Union{Int,Vector{Int}}, H::Int, d::Vector{Int},...)
Function to train multiple attention models at the same time. The function takes as input a vector of tuples, each containing the MSA and the weights of a protein family. \n 
The function trains the models for the specified number of epochs and returns a structure out::OutAttMulti containing the (Qs, Ks, V) for the different families so that the V is shared among all families. 
Keyword arguments:\n

    n_epochs:  single integer shared across all families OR a vector of integers defining how long to train on each family 
    H: the number of heads, shared across all families
    d: vector of inner dimensions, each of which is family dependent 
    savefile: path to the log file pf the training, default nothing

# Examples
```
julia> out = multi_trainer([(Z1,W1),(Z2,W2)],100,32,[23,42]);
julia> out.m
julia> out.m.Qs
julia> out.m.Ks
julia> out.m.V
```
"""
function multi_trainer(D::Vector{Tuple{Matrix{Int}, Vector{Float64}}}, n_epochs::Union{Int,Vector{Int}}, H::Int, d::Vector{Int};
    init_m = Nothing,
    η = 0.005, 
    n_batches = 50, 
    init = rand,
    λ::Union{Float64,Vector{Float64}}=fill(0.001,length(D)), 
    verbose = true,
    savefile::Union{String, Nothing} = nothing)

    NF = length(D)
    if typeof(λ) == Float64 
        λ = fill(λ,NF)
    end

    if typeof(n_epochs) == Int
        n_epochs = fill(n_epochs, NF)
    end

    #controlli vari
    length(d) == NF || error("Wrong number of d values")

    #creazione arvar per ogni famiglia
    q = maximum(D[1][1])
    Ns = zeros(Int, NF)
    Ms = zeros(Int, NF)
    batch_sizes = zeros(Int, NF)
    for i in 1:NF
        Ns[i], Ms[i] = size(D[i][1])
        batch_sizes[i] = Int(round(Ms[i]/n_batches))
    end

    m = if init_m !== Nothing
        init_m
    else
        (Qs = init.(H,d,Ns), Ks = init.(H,d,Ns), V = init(H,q,q))
    end

    t = setup(Adam(η), m)

    savefile !== nothing && (file = open(savefile,"a"))

    loaders = Vector{Any}(undef, NF)
    for i in 1:maximum(n_epochs)
        flags = i .<= n_epochs
        for n in 1:NF
            loaders[n] = DataLoader(D[n], batchsize = batch_sizes[n], shuffle = true)
        end
        loader = zip(loaders...)
        for pf in loader
            ws = [pf[m][2]/sum(pf[m][2]) for m in 1:NF]
            Zs = [pf[m][1] for m in 1:NF] 
            g = gradient(x->multi_loss(x.Qs[flags], x.Ks[flags], x.V, Zs[flags], ws[flags], λ = λ[flags], ar = false), m)[1]
            update!(t,m,g)
        end


        losses = [round(loss(m.Qs[i], m.Ks[i], m.V, D[i][1], D[i][2],  λ=λ[i]),digits=5) for i in 1:NF]
        if verbose
            print("Epoch $i") 
            [print(" loss PF$n = $(losses[n])") for n in 1:NF]
            println(" -> Total loss = $(round(sum(losses),digits=5))")
        end
        savefile !== nothing && print(file, "Epoch $i")
        savefile !== nothing && [print(file, " loss PF$n = $(losses[n])") for n in 1:NF]  
        savefile !== nothing && print(file, " -> Total loss = $(round(sum(losses),digits=5))\n")
    end

    savefile !== nothing && close(file)
    return AttentionDCA.AttOutMulti(m, nothing)


end

"""
    multi_trainer(filenames::Vector{String}, n_epochs::Union{Int,Vector{Int}}, H::Int, d::Vector{Int},...)
Function to train multiple attention models at the same time. The function takes as input a vector of filenames, each containing the fasta file with the MSA of a protein family. 
# Examples
```
julia> out = multi_trainer(["file1.fasta","file2.fasta"],100,32,[23,42]);
julia> out.m
julia> out.m.Qs
julia> out.m.Ks
julia> out.m.V
```

"""
function multi_trainer(filenames::Vector{String}, n_epochs::Union{Int,Vector{Int}}, H::Int, d::Vector{Int};
    theta::Union{Symbol,Real}=:auto,
    max_gap_fraction::Real=0.9,
    remove_dups::Bool=true,
    verbose = true,
    kwds...)

    Nf = length(filenames)
    Ws = []
    Zs = []
    Ns = zeros(Int, Nf) 

    for i in 1:Nf
        _W, _Z, Ns[i], _, _ = ReadFasta(filenames[i], max_gap_fraction, theta, remove_dups, verbose = verbose)
        push!(Zs, _Z)
        push!(Ws, _W)
    end

    data = [(Zs[i],Ws[i]) for i in 1:Nf]

    multi_trainer(data, n_epochs, H, d; verbose = verbose, kwds...)

end




"""
    stat_multi_trainer(filenames::Vector{String}, n_sim::Int, H, d,...)
Multiple iterations version of ‘multi_trainer‘. See ‘stat_trainer‘ and ‘multi_trainer‘ for more details.
# Examples
```
julia> out = stat_multi_trainer(["file1.fasta","file2.fasta"],100,32,[23,42]);
julia> out.score[1]
julia> out.score[2]

```

"""
function stat_multi_trainer(filenames::Vector{String}, n_sim::Int, H, d;
    n_epochs = 100,
    verbose = true,
    kwds...)
    D = AttentionDCA.quickread.(filenames)
    s = [[] for _ in eachindex(filenames)]
    for _ in 1:n_sim
        out = multi_trainer(D, n_epochs, H, d; verbose = verbose, kwds...)
        for i in eachindex(filenames)
            push!(s[i],score(out.m.Qs[i],out.m.Ks[i],out.m.V))
        end
    end
    for i in eachindex(filenames)
        s[i] = vcat(s[i]...)
        s[i] = unique(x->x[1:2],sort(s[i], by = x -> x[3], rev = true))
    end
    return AttentionDCA.AttOutMulti(nothing, Vector{Tuple{Int64, Int64, Float64}}.(s))
end


"""
    multi_artrainer(D::Vector{Tuple{Matrix{Int}, Vector{Float64}}}, n_epochs::Union{Int,Vector{Int}}, H::Int, d::Vector{Int}, idxperm::Vector{Vector{Int}},...)
Autoregressive version of ‘multi_trainer‘. See ‘multi_trainer‘ for more details.
# Examples
```
julia> out = multi_artrainer([(Z1,W1),(Z2,W2)],100,32,[23,42]);
julia> out[1] #vector of ArNet structures, one for each family
julia> out[2] #vector of ArVar structures, one for each family
julia> out[3] #vector of m=(Q,K,V), one for each family
julia> out[3][i].Q #query matrix for the "i-th" family
julia> out[3][i].K
julia> out[3][i].V
```
"""
function multi_artrainer(D::Vector{Tuple{Matrix{Int}, Vector{Float64}}}, n_epochs::Union{Int,Vector{Int}}, H::Int, d::Vector{Int}, idxperm::Vector{Vector{Int}};
    init_m = Nothing,
    η = 0.005, 
    n_batches = 50, 
    init = rand,
    λ::Union{Float64,Vector{Float64}}=fill(0.001,length(D)), 
    verbose = true,
    savefile::Union{String, Nothing} = nothing)

    NF = length(D)
    if typeof(λ) == Float64 
        λ = fill(λ,NF)
    end

    if typeof(n_epochs) == Int
        n_epochs = fill(n_epochs, NF)
    end

    #Various checks
    length(d) == NF || error("Wrong number of d values")
    length(idxperm) == NF || error("Wrong number of idxperm arrays")

    #Creation of ArVar for each family
    q = maximum(D[1][1])
    Ns = zeros(Int, NF)
    Ms = zeros(Int, NF)
    batch_sizes = zeros(Int, NF)
    arvars = []
    for i in 1:NF
        Ns[i], Ms[i] = size(D[i][1])
        batch_sizes[i] = Int(round(Ms[i]/n_batches))
        push!(arvars, ArVar(Ns[i],Ms[i],q,λ[i],0.0,D[i][1],D[i][2],1/Ms[i],idxperm[i]))
    end

    m = if init_m !== Nothing
        init_m
    else
        (Qs = init.(H,d,Ns), Ks = init.(H,d,Ns), V = init(H,q,q))
    end

    t = setup(Adam(η), m)

    savefile !== nothing && (file = open(savefile,"a"))

    loaders = Vector{Any}(undef, NF)
    for i in 1:maximum(n_epochs)
        flags = i .<= n_epochs
        for n in 1:NF
            loaders[n] = DataLoader(D[n], batchsize = batch_sizes[n], shuffle = true)
        end
        loader = zip(loaders...)
        for pf in loader
            ws = [pf[m][2]/sum(pf[m][2]) for m in 1:NF]
            Zs = [pf[m][1] for m in 1:NF] 
            g = gradient(x->multi_loss(x.Qs[flags], x.Ks[flags], x.V, Zs[flags], ws[flags], λ = λ[flags]),m)[1]
            
            update!(t,m,g)
        end


        losses = [round(arloss(m.Qs[i], m.Ks[i], m.V, D[i][1], D[i][2],  λ=λ[i]),digits=5) for i in 1:NF]
        if verbose
            print("Epoch $i ") 
            [print("loss PF$n = $(losses[n]), ") for n in 1:NF]
            println("-> Total loss = $(round(sum(losses),digits=5))")
        end
        savefile !== nothing && println(file, "Total loss = $(round(sum(losses),digits=5)))")    
    end
    savefile !== nothing && close(file)

    arnets = [ArNet(idxperm[i], compute_p0_J_F(D[i], m.Qs[i], m.Ks[i], m.V)...) for i in 1:NF]
 
    return arnets, arvars, m 

end


function multi_artrainer(filenames::Vector{String}, n_epochs::Union{Int,Vector{Int}}, H::Int, d::Vector{Int};
    permorder = [:NATURAL for i in eachindex(filenames)],
    theta::Union{Symbol,Real}=:auto,
    max_gap_fraction::Real=0.9,
    remove_dups::Bool=true,
    verbose = true,
    kwds...)

    Nf = length(filenames)
    Ws = []
    Zs = []
    Ns = zeros(Int, Nf) 

    for i in 1:Nf
        _W, _Z, Ns[i], _, _ = ReadFasta(filenames[i], max_gap_fraction, theta, remove_dups, verbose = verbose)
        push!(Zs, _Z)
        push!(Ws, _W)
    end

    idxperm = [check_permorder(permorder[i], Zs[i], Ws[i]) for i in 1:Nf]

    data = [(Zs[i],Ws[i]) for i in 1:Nf]

multi_artrainer(data, n_epochs, H, d, idxperm; verbose = verbose, kwds...)

end

function multi_loss(Qs, Ks, V, Zs, Ws; λ = λ, ar = true)

    if ar == true 
        foo = arloss
    else 
        foo = loss  
    end

    Nf = length(Qs)

    tot_loss = 0.0
    for i in 1:Nf
        tot_loss = tot_loss + foo(Qs[i], Ks[i], V, Zs[i], Ws[i], λ=λ[i])
    end

    return tot_loss
end

function check_permorder(permorder,Z,W) 
    N = size(Z,1)

    idxperm = if typeof(permorder) == Symbol
        if permorder === :NATURAL
            collect(1:N)
        elseif permorder === :ENTROPIC
            S = entropy(Z,W)
            sortperm(S)
        elseif permorder === :REV_ENTROPIC
            S = entropy(Z,W)
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

    return idxperm

end

function compute_p0_J_F(D, Q, K, V)
    N = size(Q,3)
    q = size(V,3)
    p0 = computep0(D)
    @tullio W[h, i, j] := Q[h,d,i]*K[h,d,j]
    W = softmax(W,dims=3) 
    @tullio J[i,j,a,b] := W[h,i,j]*V[h,a,b]*(j<i)
    J_reshaped = AttentionDCA.reshapetensor(J,N,q)
    F = [zeros(q) for _ in 1:N-1]

    return p0, J_reshaped, F
end

