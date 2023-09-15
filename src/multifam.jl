#Create a function that, given PF1 and PF2, learns (Q1,K1), (Q2,K2) and V. 
#V must be common to the two families


function multi_artrainer(D1::Tuple{Matrix{Int}, Vector{Float64}},D2::Tuple{Matrix{Int}, Vector{Float64}}, n_epochs::Int, H::Int, d1::Int, d2::Int, idxperm1::Vector{Int}, idxperm2::Vector{Int}; 
    init_m = Nothing,
    η = 0.005, 
    n_batches = 50, 
    init = rand,
    λ=0.001, 
    savefile::Union{String, Nothing} = nothing)
    


    N1,M1 = size(D1[1])
    N2,M2 = size(D2[1])
    q = maximum(D1[1])

    batch_size1 = Int(round(M1/n_batches)) 
    batch_size2 = Int(round(M2/n_batches))

    arvar1 = ArVar(N1,M1,q,λ,0.0,D1[1],D1[2],idxperm1)
    arvar2 = ArVar(N2,M2,q,λ,0.0,D2[1],D2[2],idxperm2)


    m = if init_m !== Nothing
        init_m
    else
        (Q1 = init(H,d1,N1), K1 = init(H,d1,N1), Q2 = init(H,d2,N2), K2 = init(H,d2,N2), V = init(H,q,q))
    end

    t = setup(Adam(η), m)

    savefile !== nothing && (file = open(savefile,"a"))
    

    for i in 1:n_epochs
        loader1 = DataLoader(D1, batchsize = batch_size1, shuffle = true)
        loader2 = DataLoader(D2, batchsize = batch_size2, shuffle = true)
        loader = zip(loader1, loader2)
        for (pf1,pf2) in loader
            w1 = pf1[2]/sum(pf1[2])
            w2 = pf2[2]/sum(pf2[2])
            g = gradient(x->arloss(x.Q1, x.K1, x.V, pf1[1], w1, λ=λ) + arloss(x.Q2, x.K2, x.V, pf2[1], w2, λ=λ),m)[1];
            update!(t,m,g)
        end

        loss1 = round(arloss(m.Q1, m.K1, m.V, D1[1], D1[2],  λ=λ),digits=5) 
        loss2 = round(arloss(m.Q2, m.K2, m.V, D2[1], D2[2],  λ=λ),digits=5)

        

        println("Epoch $i loss PF1 = $loss1, loss PF2 = $loss2 ---> total loss = $(round(loss1 + loss2,digits=5))")
        
        savefile !== nothing && println("Epoch $i loss PF1 = $loss1, loss PF2 = $loss2 ---> total loss = $(loss1+loss2)")
        
    end

    savefile !== nothing && close(file)
    
    p0 = computep0(D1)
    @tullio W[h, i, j] := m.Q1[h,d,i]*m.K1[h,d,j]
    W = softmax(W,dims=3) 
    @tullio J[i,j,a,b] := W[h,i,j]*m.V[h,a,b]*(j<i)
    J_reshaped = AttentionBasedPlmDCA.reshapetensor(J,N1,q)
    F = [zeros(q) for _ in 1:N1-1]
    
    arnet1 = ArNet(idxperm1, p0, J_reshaped,F)

    p0 = computep0(D2)
    @tullio W[h, i, j] := m.Q2[h,d,i]*m.K2[h,d,j]
    W = softmax(W,dims=3) 
    @tullio J[i,j,a,b] := W[h,i,j]*m.V[h,a,b]*(j<i)
    J_reshaped = AttentionBasedPlmDCA.reshapetensor(J,N2,q)
    F = [zeros(q) for _ in 1:N2-1]
    
    arnet2 = ArNet(idxperm2, p0, J_reshaped,F)


    return arnet1, arnet2, arvar1, arvar2, m
end


function multi_artrainer(filename1::String, filename2::String, n_epochs::Int, H, d1, d2;
    permorder1::Union{Symbol, Vector{Int}} = :NATURAL,
    permorder2::Union{Symbol, Vector{Int}} = :NATURAL, 
    theta::Union{Symbol,Real}=:auto,
    max_gap_fraction::Real=0.9,
    remove_dups::Bool=true,
    kwds...)
    
    time = @elapsed Weights1, Z1, N1, M1, q = ReadFasta(filename1, max_gap_fraction, theta, remove_dups)
    time = @elapsed Weights2, Z2, N2, M2, q = ReadFasta(filename2, max_gap_fraction, theta, remove_dups)
    


    idxperm1 = if typeof(permorder1) == Symbol
        if permorder1 === :NATURAL
            collect(1:N1)
        elseif permorder1 === :ENTROPIC
            S = entropy(Z1,Weights1)
            sortperm(S)
        elseif permorder1 === :REV_ENTROPIC
            S = entropy(Z1,Weights1)
            sortperm(S,rev=true)
        elseif permorder1 === :RANDOM
            randperm(N1)
        else
            error("the end of the world has come")
        end
    elseif typeof(permorder1) <: Vector
        (length(permorder1) != N1) && error("length permorder ≠ $N1")
        isperm(permorder1) && (permorder1)
    else
        error("permorder can only be a Symbol or a Vector")
    end

    idxperm2 = if typeof(permorder2) == Symbol
        if permorder2 === :NATURAL
            collect(1:N2)
        elseif permorder2 === :ENTROPIC
            S = entropy(Z2,Weights2)
            sortperm(S)
        elseif permorder2 === :REV_ENTROPIC
            S = entropy(Z2,Weights2)
            sortperm(S,rev=true)
        elseif permorder2 === :RANDOM
            randperm(N2)
        else
            error("the end of the world has come")
        end
    elseif typeof(permorder2) <: Vector
        (length(permorder2) != N2) && error("length permorder ≠ $N2")
        isperm(permorder2) && (permorder2)
    else
        error("permorder can only be a Symbol or a Vector")
    end


    #ArDCA.permuterow!(Z,idxperm) 
    data1 = (Z1,Weights1)
    data2 = (Z2,Weights2)
    println("preprocessing took $time seconds")
    multi_artrainer(data1, data2, n_epochs, H, d1, d2, idxperm1, idxperm2; kwds...)
end