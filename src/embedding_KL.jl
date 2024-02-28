function one_hot(msa::Array{Int,2}; q::Int = maximum(msa))
    N, M = size(msa)
    new_msa = zeros(q,N,M)
    
    @tullio indexes[i,m] := msa[i,m]
    @tullio res[indexes[i,m],i,m] := new_msa[indexes[i,m],i,m] + 1

    return res

end




# function compute_attention(m::N ,Z::Matrix{T1}) where {T1<: Integer}
    
#     Z_one_hot = one_hot(Z) #L,q,M
    

#     @tullio A[m,i,j] := Q[i,d,m]*K[j,d,m]
#     A = softmax_notinplace(A, dims = 3)
#     @tullio Y[d,i,m] := A[m,i,j]*V[j,d,m]

#     return softmax(Y,dims=2)
# end

function log0(x)
    if x > 0
        return log(x)
    else
        return 0.0
    end
end

# function my_loss(m::NamedTuple{(:Wq, :Wk, :Wv)}, Z::Matrix{T1}, weigths::Vector{T2}; H = 2, q = maximum(Z)) where {T1 <: Integer, T2 <: Real}
#     N = size(Z,1)
#     M = size(Z,2)
    
#     H,de,d = size(m.Wq)
#     Zoh = one_hot(Z) #q,L,M

#     size(Zoh,1) == de || error("The size of the one hot encoding is not the same as the first dimension of Wq")

#     @tullio Q[h,i,d2,m] := Zoh[d1,i,m]*m.Wq[h,d1,d2]
#     @tullio K[h,i,d2,m] := Zoh[d1,i,m]*m.Wk[h,d1,d2]
#     @tullio V[h,i,d2,m] := Zoh[d1,i,m]*m.Wv[h,d1,d2]
#     @tullio A[h,i,j] := Q[h,i,d,m]*K[h,j,d,m]
#     A = softmax_notinplace(A/M, dims = 3)
#     @tullio Y[h,d,i,m] := A[h,i,j]*V[h,j,d,m]
#     _Y = dropdims(sum(Y, dims=1),dims=1) / H
#     Y = softmax_notinplace(_Y, dims = 1)
#     Zoh = reshape(Zoh, q*N, M) #qL,M
#     Y = reshape(Y, q*N, M) #qL,M
#     @tullio loss[m] := Zoh[l,m]*log(Y[l,m])
#     return -loss'*weigths
# end

function my_loss(m::NamedTuple{(:Wq, :Wk, :Wv)}, Z::Matrix{T1}, weigths::Vector{T2}; λ=0.001, reg = :m) where {T1 <: Integer, T2 <: Real}
    N = size(Z,1)
    M = size(Z,2)
    
    reg ∈ [:m, :att] || error("The regularization method is not valid, only :m and :att are allowed.")

    de,d,H = size(m.Wq)
    Zoh = one_hot(Z) #q,L,M

    size(Zoh,1) == de || error("The size of the one hot encoding is not the same as the first dimension of Wq")

    @tullio Q[i,d2,m,h] := Zoh[d1,i,m]*m.Wq[d1,d2,h]
    @tullio K[i,d2,m,h] := Zoh[d1,i,m]*m.Wk[d1,d2,h]
    @tullio V[i,d2,m,h] := Zoh[d1,i,m]*m.Wv[d1,d2,h]
    @tullio A[i,j,h] := Q[i,d,m,h]*K[j,d,m,h]
    A = softmax_notinplace(A/M, dims = 2)
    @tullio Y[d,i,m,h] := A[i,j,h]*V[j,d,m,h]
    @tullio _Y[d,i,m] := Y[d,i,m,h]
    Y = softmax_notinplace(_Y/H, dims = 1)
    @tullio loss := Zoh[d,i,m] * log(Y[d,i,m]) * weigths[m]
    
    if reg == :m
        return -loss + λ*(sum(abs2,m.Wq) + sum(abs2,m.Wk) + sum(abs2,m.Wv))
    elseif reg == :att
        return -loss + λ*sum(abs2,A)
    end
end

function my_trainer(D::Tuple{Matrix{T1}, Vector{T2}},n_epochs::Int; 
    H::Int = 32,
    d::Int = 21,
    q::Int = 21,
    batch_size::Int = 1000,
    η::Float64 = 0.005,
    λ::Float64 = 0.001,
    reg::Symbol = :m,
    init_m = nothing, 
    init_fun = rand, 
    structfile::String = "precompilation_data/PF00014_struct.dat",
    verbose = true,
    savefile::Union{String, Nothing} = nothing) where {T1<:Integer, T2<:Real}
    
    N,_ = size(D[1])

    m = if init_m !== nothing
        init_m
    else
        (Wq = init_fun(q,d,H), Wk = init_fun(q,d,H), Wv = init_fun(q,q,H))
    end
    t = setup(Adam(η), m)

    savefile !== nothing && (file = open(savefile,"a"))

    for i in 1:n_epochs
        loader = DataLoader(D, batchsize = batch_size, shuffle = true)
        for (z,w) in loader
            _w = w/sum(w)
            g = gradient(x->my_loss(x, z, _w, λ = λ, reg = reg),m)[1];
            update!(t,m,g)
        end

        PPV = ppv_attention(m,D[1],structfile)
        l = round(my_loss(m, D[1], D[2], λ = λ, reg = reg),digits=5) 
        p = round((PPV[N]),digits=3)
        verbose && println("Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
        savefile !== nothing && println(file, "Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
    end

    savefile !== nothing && close(file)
    return m
end

function ppv_attention(m::NamedTuple{(:Wq, :Wk, :Wv)}, Z::Matrix{T1}, structfile::String; version = mean, APC = true) where {T1<:Integer}
    
    Zoh = one_hot(Z) #q,L,M
    M = size(Z,2)

    # @tullio Q[h,i,d2,m] := Zoh[d1,i,m]*m.Wq[h,d1,d2]
    # @tullio K[h,i,d2,m] := Zoh[d1,i,m]*m.Wk[h,d1,d2]

    # @tullio A[h,i,j] := Q[h,i,d,m]*K[h,j,d,m]
    # A = softmax_notinplace(A/M, dims = 3)

    @tullio Q[i,d2,m,h] := Zoh[d1,i,m]*m.Wq[d1,d2,h]
    @tullio K[i,d2,m,h] := Zoh[d1,i,m]*m.Wk[d1,d2,h]
    @tullio A[i,j,h] := Q[i,d,m,h]*K[j,d,m,h]
    A = softmax_notinplace(A/M, dims = 2)


    Am = version(A, dims = 3)[:,:,1]
    Am = (Am + Am')/2
    if APC 
        Am = AttentionDCA.correct_APC(Am)
    end
    
    s = score_from_matrix(Am)

    PPV = compute_PPV(s,structfile)
    return PPV
end

score_from_matrix(A) = sort([(j,i,A[j,i]) for i in 2:size(A,2) for j in 1:i-1], by = x->x[3], rev = true)

# function my_loss(m,Z,weights; H = 2, λ=0.001,verbose=false)

#     N = size(Z,1)
#     M = size(Z,2)
#     q = maximum(Z)
#     Z_one_hot = one_hot(Z, q=21) #q,L,M
    
#     #         L,d,M                de,L,M       de,d
#     @tullio Q[h, i,d2,m] := Z_one_hot[d1,i,m]*m.Wq[h, d1,d2]
#     @tullio K[h, i,d2,m] := Z_one_hot[d1,i,m]*m.Wk[h, d1,d2]
#     #         L,de,M                de,L,M       de,de
#     @tullio V[h, i,d2,m] := Z_one_hot[d1,i,m]*m.Wv[h, d1,d2]

#     @tullio A[h, m,i,j] := Q[h,i,d,m]*K[h,j,d,m]
#     A = softmax_notinplace(sum(A,dims=2)[:,1,:,:]/M, dims =3) #L,L
#     #         de,L,M    M,L,       L,de,M
#     @tullio Y[h,d,i,m] := A[h,i,j]*V[h,j,d,m]

#     Y = softmax_notinplace(Y,dims=2)


#     @tullio KL_div[h,d,i,m] := Z_one_hot[d,i,m]*log(Y[h,d,i,m])
#     @tullio loss[h,d,i] := KL_div[h,d,i,m]*weights[m]
#     verbose && println("loss: $(-sum(loss)), regQ: $(λ*sum(abs2,m.Wq)), regK: $(λ*sum(abs2,m.Wk)), regV: $(λ*sum(abs2,m.Wv))")
#     return -sum(loss) + λ*sum(abs2,m.Wq) + λ*sum(abs2,m.Wk) + λ*sum(abs2,m.Wv)
    
# end


# function my_trainer(D::Tuple{Matrix{T1}, Vector{T2}},n_epochs::Int;
#     H = 2,
#     d::Int = 256,
#     batch_size::Int = 1000,
#     η::Float64 = 0.005,
#     λ::Float64 = 0.001,
#     init_m = nothing, 
#     init_fun = rand, 
#     structfile::String = "precompilation_data/PF00014_struct.dat",
#     verbose = true,
#     savefile::Union{String, Nothing} = nothing) where {T1<:Integer, T2<:Real}
    
#     N,_ = size(D[1])
#     q = maximum(D[1])

#     m = if init_m !== nothing
#         init_m
#     else
#         (Wq = init_fun(H,q,d), Wk = init_fun(H,q,d), Wv = init_fun(H,q,q))
#     end
#     t = setup(Adam(η), m)

#     savefile !== nothing && (file = open(savefile,"a"))
#     for i in 1:n_epochs
#         loader = DataLoader(D, batchsize = batch_size, shuffle = true)
#         for (z,w) in loader
#             _w = w/sum(w)
#             g = gradient(x->my_loss(x, z, _w, H = H),m)[1];
#             update!(t,m,g)
#         end
#         print("epoch $i ")
#         my_loss(m,D[1],D[2],H=H,verbose=true)

#         # s = score(m.Q,m.K,m.V);
#         # PPV = compute_PPV(s,structfile)
#         #l = round(loss(m.Q, m.K, m.V, D[1], D[2], λ = λ),digits=5) 
#         # p = round((PPV[N]),digits=3)
#         # verbose && println("Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
#         # savefile !== nothing && println(file, "Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
#     end

#     savefile !== nothing && close(file)
#     return m
# end


# function my_loss(Wq::Array{T1, 3},
#     Wk::Array{T1, 3},
#     Wv::Array{T1, 3}, 
#     Z::Matrix{T2},
#     weights::Vector{T3};
#     λ::Float64 = 0.001) where {T1<:Real, T2<: Integer, T3<:Real}

#     q,L,M = size(Z)

#     Zoh = one_hot(Z) #q,L,M
#     @tullio Q[h, i,d2,m] := Zoh[d1,i,m]*Wq[h, d1,d2]
#     @tullio K[h, i,d2,m] := Zoh[d1,i,m]*Wk[h, d1,d2]
#     @tullio Vm[h, i,d2,m] := Zoh[d1,i,m]*Wv[h, d1,d2]

#     @tullio V[h,i,d] := Vm[h,i,d,m]

#     #compute softmax function of Q*K'
#     @tullio sf[i, j, h, m] := Q[h,d,i,m]*K[h,d,j,m]
#     sf = softmax_notinplace(sum(sf, dims=4)[:,:,:,1]/M,dims=2) 
    
#     #compute J tensor
#     @tullio J[i,j,a,b] := sf[i,j,h]*V[h,a,b]*(j!=i)

#     #compute the energy of the sequences and the partition function
#     @tullio mat_ene[a,r,m] := J[r,j,a,Z[j,m]]
#     lge = logsumexp(mat_ene,dims=1)[1,:,:]

#     @tullio pl = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
#     pl = -1*pl
    
#     #compute the loss
#     reg = λ*(sum(abs2, J))
    
#     pl = pl + reg

#     return pl
# end 

# my_closs(m::NamedTuple{(:Q, :K, :V)}, Z::Matrix{T2}, weights::Vector{T1}; kwds...) where {T1<: Real, T2 <: Integer} = my_loss(m..., Z, weights; kwds...) 

# function my_trainer(D::Tuple{Matrix{T1}, Vector{T2}},n_epochs::Int; 
#     H::Int = 32,
#     d::Int = 23,
#     batch_size::Int = 1000,
#     η::Float64 = 0.005,
#     λ::Float64 = 0.001,
#     init_m = nothing, 
#     init_fun = rand, 
#     structfile::String = "precompilation_data/PF00014_struct.dat",
#     verbose = true,
#     savefile::Union{String, Nothing} = nothing) where {T1<:Integer, T2<:Real}
    
#     N,_ = size(D[1])
#     q = maximum(D[1])

#     m = if init_m !== nothing
#         init_m
#     else
#         (Wq = init_fun(H,q,d), Wk = init_fun(H,q,d), Wv = init_fun(H,q,q))
#     end
#     t = setup(Adam(η), m)

#     savefile !== nothing && (file = open(savefile,"a"))
    
#     for i in 1:n_epochs
#         loader = DataLoader(D, batchsize = batch_size, shuffle = true)
#         for (z,w) in loader
#             _w = w/sum(w)
#             g = gradient(x->my_loss(x.Wq, x.Wk, x.Wv, z, _w, λ = λ),m)[1];
#             update!(t,m,g)
#         end

#         #s = score(m.Q,m.K,m.V);
#         #PPV = compute_PPV(s,structfile)
#         l = round(my_loss(m.Wq, m.Wk, m.Wv, D[1], D[2], λ = λ),digits=5) 
#         p = round((PPV[N]),digits=3)
#         verbose && println("Epoch $i loss = $l")# \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
#         #savefile !== nothing && println(file, "Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
#     end

#     savefile !== nothing && close(file)
#     return m
# end

# function my_trainer(filename::String, n_epochs::Int;
#     theta::Union{Symbol,Real}=:auto,
#     max_gap_fraction::Real=0.9,
#     remove_dups::Bool=true,
#     verbose = true,
#     kwds...)
    
#     time = @elapsed Weights, Z, _, _, _ = ReadFasta(filename, max_gap_fraction, theta, remove_dups, verbose = verbose)
    
#     data = (Z,Weights)
#     verbose && println("preprocessing took $time seconds")
#     my_trainer(data, n_epochs; verbose = verbose, kwds...)
# end