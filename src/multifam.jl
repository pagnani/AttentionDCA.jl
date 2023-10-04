#Create a function that, given PF1 and PF2, learns (Q1,K1), (Q2,K2) and V. 
#V must be common to the two families

function multi_trainer(D::Vector{Tuple{Matrix{Int}, Vector{Float64}}}, n_epochs::Union{Int,Vector{Int}}, H::Int, d::Vector{Int};
    init_m = Nothing,
    η = 0.005, 
    n_batches = 50, 
    init = rand,
    λ::Union{Float64,Vector{Float64}}=fill(0.001,length(D)), 
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
            g = gradient(x->multi_loss(x.Qs[flags], x.Ks[flags], x.V, Zs[flags], ws[flags], λ = λ[flags], ar = false),m)[1]
            #if sum(flags) != 0.0
            #    g.Qs[flags] = zeros.(H,d[flags], Ns[flags])
            #    g.Ks[flags] = zeros.(H,d[flags], Ns[flags])
            #end
            update!(t,m,g)
            #println(sum(m.Qs[1]),"   ",sum(m.Ks[1]),"   ",sum(m.Qs[2]),"   ",sum(m.Ks[2]),"   ", sum(m.V))
        end


        losses = [round(loss(m.Qs[i], m.Ks[i], m.V, D[i][1], D[i][2],  λ=λ[i]),digits=5) for i in 1:NF]
        print("Epoch $i ") 
        [print("loss PF$n = $(losses[n]), ") for n in 1:NF]
        println("-> Total loss = $(round(sum(losses),digits=5))")
        
        savefile !== nothing && println("total loss = $(round(sum(losses),digits=5)))")    
    end
    savefile !== nothing && close(file)

    return m


end


function multi_trainer(filenames::Vector{String}, n_epochs::Union{Int,Vector{Int}}, H::Int, d::Vector{Int};
    theta::Union{Symbol,Real}=:auto,
    max_gap_fraction::Real=0.9,
    remove_dups::Bool=true,
    kwds...)

    Nf = length(filenames)
    Ws = []
    Zs = []
    Ns = zeros(Int, Nf) 

    for i in 1:Nf
        _W, _Z, Ns[i], _, _ = ReadFasta(filenames[i], max_gap_fraction, theta, remove_dups)
        push!(Zs, _Z)
        push!(Ws, _W)
    end

    data = [(Zs[i],Ws[i]) for i in 1:Nf]

    multi_trainer(data, n_epochs, H, d; kwds...)

end



function multi_artrainer(D::Vector{Tuple{Matrix{Int}, Vector{Float64}}}, n_epochs::Union{Int,Vector{Int}}, H::Int, d::Vector{Int}, idxperm::Vector{Vector{Int}};

    init_m = Nothing,
    η = 0.005, 
    n_batches = 50, 
    init = rand,
    λ::Union{Float64,Vector{Float64}}=fill(0.001,length(D)), 
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
    length(idxperm) == NF || error("Wrong number of idxperm arrays")

    #creazione arvar per ogni famiglia
    q = maximum(D[1][1])
    Ns = zeros(Int, NF)
    Ms = zeros(Int, NF)
    batch_sizes = zeros(Int, NF)
    arvars = []
    for i in 1:NF
        Ns[i], Ms[i] = size(D[i][1])
        batch_sizes[i] = Int(round(Ms[i]/n_batches))
        push!(arvars, ArVar(Ns[i],Ms[i],q,λ[i],0.0,D[i][1],D[i][2],idxperm[i]))
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
            #if sum(flags) != 0.0
            #    g.Qs[flags] = zeros.(H,d[flags], Ns[flags])
            #    g.Ks[flags] = zeros.(H,d[flags], Ns[flags])
            #end
            update!(t,m,g)
            #println(sum(m.Qs[1]),"   ",sum(m.Ks[1]),"   ",sum(m.Qs[2]),"   ",sum(m.Ks[2]),"   ", sum(m.V))
        end


        losses = [round(arloss(m.Qs[i], m.Ks[i], m.V, D[i][1], D[i][2],  λ=λ[i]),digits=5) for i in 1:NF]
        print("Epoch $i ") 
        [print("loss PF$n = $(losses[n]), ") for n in 1:NF]
        println("-> Total loss = $(round(sum(losses),digits=5))")
        
        savefile !== nothing && println("total loss = $(round(sum(losses),digits=5)))")    
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
    kwds...)

    Nf = length(filenames)
    Ws = []
    Zs = []
    Ns = zeros(Int, Nf) 

    for i in 1:Nf
        _W, _Z, Ns[i], _, _ = ReadFasta(filenames[i], max_gap_fraction, theta, remove_dups)
        push!(Zs, _Z)
        push!(Ws, _W)
    end

    idxperm = [check_permorder(permorder[i], Zs[i], Ws[i]) for i in 1:Nf]

    data = [(Zs[i],Ws[i]) for i in 1:Nf]

    multi_artrainer(data, n_epochs, H, d, idxperm; kwds...)

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
    J_reshaped = AttentionBasedPlmDCA.reshapetensor(J,N,q)
    F = [zeros(q) for _ in 1:N-1]

    return p0, J_reshaped, F
end