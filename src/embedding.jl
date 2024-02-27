# import Flux: train!, params, ADAM

# # struct attention{T1 <: Real}
# #     Wq::Array{T1, 2}
# #     Wk::Array{T1, 2}
# #     Wv::Array{T1, 2}
# # end

# # function one_hot(msa::Array{Int,2})
# #     N, M = size(msa)
# #     q = maximum(msa)
# #     new_msa = zeros(q,N,M)
    
# #     for i in 1:N
# #         for j in 1:M
# #             index = msa[i,j]  
# #             new_msa[index, i, j] = new_msa[index, i, j] + 1
# #         end
# #     end
# #     return new_msa
# # end

# function one_hot(msa::Array{Int,2}; q::Int = maximum(msa))
#     N, M = size(msa)
#     new_msa = zeros(q,N,M)
    
#     @tullio indexes[i,m] := msa[i,m]
#     @tullio res[indexes[i,m],i,m] := new_msa[indexes[i,m],i,m] + 1

#     return res

# end




# # function (attention::attention)(Z::Matrix{T1}) where {T1<: Integer}
    
# #     Z_one_hot = one_hot(Z) #L,q,M
# #     @tullio Q[i,d2,m] := Z_one_hot[d1,i,m]*attention.Wq[d1,d2]
# #     @tullio K[i,d2,m] := Z_one_hot[d1,i,m]*attention.Wk[d1,d2]
# #     @tullio V[i,d2,m] := Z_one_hot[d1,i,m]*attention.Wv[d1,d2]

# #     @tullio A[m,i,j] := Q[i,d,m]*K[j,d,m]
# #     A = softmax_notinplace(A, dims = 3)
# #     @tullio Y[d,i,m] := A[m,i,j]*V[j,d,m]

# #     return softmax(Y,dims=2)
# # end

# # function new_loss(Z::Matrix{T1}, weigths::Vector{T2}, a::attention{T2}; q = maximum(Z)) where {T1 <: Integer, T2 <: Real}
# #     N = size(Z,1)
# #     M = size(Z,2)
    
    
# #     Y = a(Z)
# #     Zoh = one_hot(Z) #q,L,M

# #     Zoh = reshape(Zoh, q*N, M) #qL,M
# #     Y = reshape(Y, q*N, M) #qL,M

# #     loss = Zoh.*log.(Y)
# #     loss = loss*weigths
# #     println(sum(loss))
# #     return sum(loss)
# # end

# # function new_trainer(D::Tuple{Matrix{T1}, Vector{T2}},n_epochs::Int; 
# #     H::Int = 32,
# #     d::Int = 23,
# #     batch_size::Int = 1000,
# #     η::Float64 = 0.005,
# #     λ::Float64 = 0.001,
# #     init_m = nothing, 
# #     init_fun = rand, 
# #     structfile::String = "precompilation_data/PF00014_struct.dat",
# #     verbose = true,
# #     savefile::Union{String, Nothing} = nothing) where {T1<:Integer, T2<:Real}
    
# #     N,_ = size(D[1])
# #     q = maximum(D[1])

# #     m = if init_m !== nothing
# #         init_m
# #     else
# #         (Q = init_fun(d,d), K = init_fun(d,d), V = init_fun(d,d))
# #     end
# #     t = setup(Adam(η), m)

# #     a = attention(m...)

# #     savefile !== nothing && (file = open(savefile,"a"))
# #     Zoi = DataLoader(D, batchsize = batch_size, shuffle = true)
# #     myloss(Z, W) = new_loss(Z, W, a)

# #     for i in 1:n_epochs
# #         # loader = DataLoader(D, batchsize = batch_size, shuffle = true)
# #         # for (z,w) in loader
# #         #     _w = w/sum(w)
# #         #     g = gradient(new_loss(z,_w,a),a.Wq,a.Wk,a.Wv)[1];
# #         #     update!(t,m,g)
# #         # end

# #         train!(
# #             myloss,
# #             params(a),
# #             Zoi,
# #             ADAM(η)
# #         )
# #         #s = score(m.Q,m.K,m.V);
# #         #PPV = compute_PPV(s,structfile)
# #         #l = round(loss(m.Q, m.K, m.V, D[1], D[2], λ = λ),digits=5) 
# #         #p = round((PPV[N]),digits=3)
# #         #verbose && println("Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
# #         #savefile !== nothing && println(file, "Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
# #     end

# #     savefile !== nothing && close(file)
# #     return m
# # end


# # function my_loss(m,Z,weights; H = 2, λ=0.001,verbose=false)

# #     N = size(Z,1)
# #     M = size(Z,2)
# #     q = maximum(Z)
# #     Z_one_hot = one_hot(Z, q=21) #q,L,M
    
# #     #         L,d,M                de,L,M       de,d
# #     @tullio Q[h, i,d2,m] := Z_one_hot[d1,i,m]*m.Wq[h, d1,d2]
# #     @tullio K[h, i,d2,m] := Z_one_hot[d1,i,m]*m.Wk[h, d1,d2]
# #     #         L,de,M                de,L,M       de,de
# #     @tullio V[h, i,d2,m] := Z_one_hot[d1,i,m]*m.Wv[h, d1,d2]

# #     @tullio A[h, m,i,j] := Q[h,i,d,m]*K[h,j,d,m]
# #     A = softmax_notinplace(sum(A,dims=2)[:,1,:,:]/M, dims =3) #L,L
# #     #         de,L,M    M,L,       L,de,M
# #     @tullio Y[h,d,i,m] := A[h,i,j]*V[h,j,d,m]

# #     Y = softmax_notinplace(Y,dims=2)


# #     @tullio KL_div[h,d,i,m] := Z_one_hot[d,i,m]*log(Y[h,d,i,m])
# #     @tullio loss[h,d,i] := KL_div[h,d,i,m]*weights[m]
# #     verbose && println("loss: $(-sum(loss)), regQ: $(λ*sum(abs2,m.Wq)), regK: $(λ*sum(abs2,m.Wk)), regV: $(λ*sum(abs2,m.Wv))")
# #     return -sum(loss) + λ*sum(abs2,m.Wq) + λ*sum(abs2,m.Wk) + λ*sum(abs2,m.Wv)
    
# # end


# # function my_trainer(D::Tuple{Matrix{T1}, Vector{T2}},n_epochs::Int;
# #     H = 2,
# #     d::Int = 256,
# #     batch_size::Int = 1000,
# #     η::Float64 = 0.005,
# #     λ::Float64 = 0.001,
# #     init_m = nothing, 
# #     init_fun = rand, 
# #     structfile::String = "precompilation_data/PF00014_struct.dat",
# #     verbose = true,
# #     savefile::Union{String, Nothing} = nothing) where {T1<:Integer, T2<:Real}
    
# #     N,_ = size(D[1])
# #     q = maximum(D[1])

# #     m = if init_m !== nothing
# #         init_m
# #     else
# #         (Wq = init_fun(H,q,d), Wk = init_fun(H,q,d), Wv = init_fun(H,q,q))
# #     end
# #     t = setup(Adam(η), m)

# #     savefile !== nothing && (file = open(savefile,"a"))
# #     for i in 1:n_epochs
# #         loader = DataLoader(D, batchsize = batch_size, shuffle = true)
# #         for (z,w) in loader
# #             _w = w/sum(w)
# #             g = gradient(x->my_loss(x, z, _w, H = H),m)[1];
# #             update!(t,m,g)
# #         end
# #         print("epoch $i ")
# #         my_loss(m,D[1],D[2],H=H,verbose=true)

# #         # s = score(m.Q,m.K,m.V);
# #         # PPV = compute_PPV(s,structfile)
# #         #l = round(loss(m.Q, m.K, m.V, D[1], D[2], λ = λ),digits=5) 
# #         # p = round((PPV[N]),digits=3)
# #         # verbose && println("Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
# #         # savefile !== nothing && println(file, "Epoch $i loss = $l \t PPV@L = $p \t First Error = $(findfirst(x->x!=1, PPV))")
# #     end

# #     savefile !== nothing && close(file)
# #     return m
# # end


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