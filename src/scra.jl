# # # function minimize_pl(alg::PlmAlg, var::AttPlmVar;
# # #     initx0::Union{Nothing, Vector{Float64}} = nothing)
# # #     @extract var : N H d q2 LL = 2*H*N*d + H*q2
    
# # #     x0 = if initx0 === nothing 
# # #         rand(Float64, LL)
# # #     else 
# # #         initx0
# # #     end
# # #     pl = 0.0
# # #     parameters = zeros(LL)

# # #     opt = Opt(alg.method, length(x0))
# # #     ftol_abs!(opt, alg.epsconv)
# # #     xtol_rel!(opt, alg.epsconv)
# # #     xtol_abs!(opt, alg.epsconv)
# # #     ftol_rel!(opt, alg.epsconv)
# # #     maxeval!(opt, alg.maxit)
# # #     min_objective!(opt, (x, g) -> optimfunwrapper(g,x, var))
# # #     elapstime = @elapsed  (minf, minx, ret) = optimize(opt, x0)
# # #     alg.verbose && @printf("pl = %.4f\t time = %.4f\t", minf, elapstime)
# # #     alg.verbose && println("exit status = $ret")
# # #     pl = minf
# # #     parameters .= minx

# # #     return parameters, pl, elapstime
# # # end

# # function scra_pl_and_grad!(grad::Vector{Float64}, x::Vector{Float64}, plmvar::AttPlmVar)
    
# #     @extract plmvar : H N M d q Z λ = N*q*lambda/M weights = W
    
# #     L = 2*H*N*d + H*q*q 
# #     L == length(x) || error("Wrong dimension of parameter vector")
# #     L == length(grad) || error("Wrong dimension of gradient vector")

# #     Q = reshape(x[1:N*d],d,N)
# #     K = reshape(x[N*d+1 : 2*H*N*d],d,N)
# #     V = reshape(x[2*N*d+1:end],q,q)

# #     pseudologlikelihood = zeros(Float64, N)
# #     reg = zeros(Float64, N)

# #     data = AttComputationQuantities(N,H,q)

     
# #     for site in 1:N 
# #         pseudologlikelihood[site], reg[site] = update_Q_site!(grad, Z, Q, K, V, site, weights, λ, data)
  
# #         update_K_site!(grad, Q, V, site, λ, data.sf, data.J, data.fact)
# #     end

# #     update_V!(grad, Q, V, λ, data)
    
# #     regularisation = sum(reg)
# #     total_pslikelihood = sum(pseudologlikelihood) + regularisation
    
# #     println(total_pslikelihood," ",regularisation)
# #     return total_pslikelihood

# # end

# # function scra_update_Q_site!(grad::Vector{Float64}, Z::Array{Int,2}, Q::Array{Float64,2}, K::Array{Float64,2}, V::Array{Float64,2}, site::Int, weights::Vector{Float64}, lambda::Float64, data::AttComputationQuantities)
# #     size(Q) == size(K) || error("Wrong dimensionality for Q and K")
# #     d,N = size(Q)
# #     q,q = size(V)
    
# #     N,M = size(Z) 
# #     @tullio W[j] := Q[d,$site]*K[d,j] #order HNd
# #     sf = softmax(W) 
# #     data.sf[site,:] .= sf

# #     @tullio J_site[j,a,b] := sf[j]*V[a,b]*(site!=j) #order HNq^2
# #     data.J[site,:,:,:] .= J_site 

# #     @tullio mat_ene[a,m] := data.J[$site,j,a,Z[j,m]] #order NMq
# #     partition = sumexp(mat_ene,dims=1) #partition function for each m ∈ 1:M 

# #     @tullio prob[a,m] := exp(mat_ene[a,m])/partition[m] #order Mq
# #     lge = log.(partition) 

# #     Z_site = view(Z,site,:) 
# #     @tullio pl = weights[m]*(mat_ene[Z_site[m],m] - lge[m]) #order M
# #     pl *= -1


# #     @tullio mat[a,b,j] := weights[m]*(Z[j,m]==b)*((Z_site[m]==a)-prob[a,m]) (a in 1:q, b in 1:q) #order NMq^2
# #     mat[:,:,site] .= 0.0 
# #     data.mat[site,:,:,:] .= mat

# #     @tullio fact[j] := mat[a,b,j]*V[a,b] #order HNq^2
# #     data.fact[site,:] .= fact 
# #     outersum = zeros(Float64, N)

# #     @inbounds for counter in (site-1)*d + 1 : site*d 
# #         y,_ = scra_counter_to_index(counter, N, d, q)
# #         @tullio  innersum = K[$y,j]*sf[j] #order N
# #         @tullio  outersum[j] = (K[$y,j]*sf[j] - sf[j]*innersum) #order N
# #         @tullio  scra = fact[j]*outersum[j] #order N
# #         @tullio  ∇reg =  J_site[j,a,b]*V[a,b]*outersum[j] #order Nq^2
# #         grad[counter] = -scra + 2*lambda*∇reg 
# #     end
# #     reg = lambda*L2Tensor(J_site) 

# #     return pl, reg
# # end

# # function update_K_site!(grad::Vector{Float64}, Q::Array{Float64,2}, V::Array{Float64,2}, site::Int, lambda::Float64, sf::Array{Float64,2}, J::Array{Float64,4}, fact::Array{Float64,2}) 
# #     d,N = size(Q)
# #     _,q,_ = size(V)
    
# #     scra = zeros(Float64,N,N)
# #     scra1 = zeros(Float64,q,q)

# #     @inbounds for counter in N*d+(site-1)*d+1:N*d + site*d
# #         y,x = scra_counter_to_index(counter, N, d, q) #h, lower dim, position
# #         @tullio scra[i,j] = Q[$y,i]*(sf[i,j]*(x==j) - sf[i,j]*sf[i,$x]) #order N^2
# #         @tullio scra2 = scra[i,j]*fact[i,j] #order N^2
# #         @tullio scra1[a,b] = scra[i,j]*J[i,j,a,b] #order N^2q^2
# #         @tullio ∇reg = scra1[a,b]*V[a,b] #order q^2
# #         grad[counter] = - scra2 + 2*lambda*∇reg
# #     end
# #     return
# # end

# # function update_V!(grad::Vector{Float64}, Q::Array{Float64,2}, V::Array{Float64,2}, lambda::Float64, data::AttComputationQuantities)

# #     d,N = size(Q)
# #     q,q = size(V)


# #     grad[2*N*d+1:2*N*d + q*q] .= 0.0
# #     scra = zeros(Float64, N)
# #     @inbounds for counter in 2*N*d+1:2*N*d + q*q
# #         a,b = scra_counter_to_index(counter, N,d,q)
# #         @tullio scra[site] = data.mat[site,$a,$b,j]*data.sf[$h,site,j] #order N^2
# #         @tullio ∇reg = data.J[i,j,$a,$b]*data.sf[$h,i,j] #order N^2
        
# #         grad[counter] = -sum(scra) + 2*lambda*∇reg
        
# #     end
# #     return
# # end


# # function scra_counter_to_index(l::Int, N::Int, d:: Int, Q::Int; verbose::Bool=false)
# #     # h::Int = 0
# #     if l <= N*d
# #         i::Int = ceil(l/d)
# #         m = l-(d)*(i-1)
# #         verbose && println("m = $m \ni = $i")
# #         return m,i
# #     elseif N*d < l <= 2*N*d 
# #         l-=d*N
# #         j::Int = ceil(l/d)
# #         n::Int = l-(d)*(j-1)
# #         verbose && println("n = $n \nj = $j")
# #         return n, j
# #     else
# #         l-=2*N*d
# #         b::Int = ceil(l/Q)
# #         a::Int = l-(Q)*(b-1)
# #         verbose && println("a = $a \nb = $b \n")
# #         return a, b
# #     end
# # end





# function attention(Z::Array{T,2},Weights::Vector{Float64};
#     H::Int = 32,
#     d::Int = 20,
#     output::Union{String,Nothing} = nothing,
#     initx0 = nothing,
#     min_separation::Int=6,
#     theta=:auto,
#     lambda::Real=0.01,
#     epsconv::Real=1.0e-5,
#     maxit::Int=1000,
#     verbose::Bool=true,
#     method::Symbol=:LD_LBFGS) where T <: Integer

#     all(x -> x > 0, Weights) || throw(DomainError("vector W should normalized and with all positive elements"))
#     isapprox(sum(Weights), 1) || throw(DomainError("sum(W) ≠ 1. Consider normalizing the vector W"))
    
#     N, M = size(Z)
#     M = length(Weights)
#     q = Int(maximum(Z))
#     plmalg = PlmAlg(method, verbose, epsconv, maxit)
#     plmvar = AttPlmVar(N, M, d, q, H, lambda, Z, Weights) #MODIFYYYY
#     # open("prova" * ".log", "w") do out
#     #     redirect_stdout(out) do
#     #         parameters, pslike, elapstime, numevals= minimize_pl(plmalg, plmvar,initx0=initx0)
#     #         Q = reshape(parameters[1:H*d*N],H,d,N)
#     #         K = reshape(parameters[H*d*N+1:2*H*d*N],H,d,N) 
#     #         V = reshape(parameters[2*H*d*N+1:end],H,q,q)
#     #         return AttPlmOut(Q, K, V, pslike), elapstime, numevals
#     #     end
#     # end
#     parameters, pslike, elapstime, numevals= minimize_pl(plmalg, plmvar,initx0=initx0)
#     Q = reshape(parameters[1:H*d*N],N,d,H)
#     K = reshape(parameters[H*d*N+1:2*H*d*N],N,d,H) 
#     V = reshape(parameters[2*H*d*N+1:end],q,q,H)
#     return AttPlmOut(Q, K, V, pslike), elapstime, numevals

# end

# function attention(filename::String;
#     theta::Union{Symbol,Real}=:auto,
#     max_gap_fraction::Real=0.9,
#     remove_dups::Bool=true,
#     kwds...)
    
#     time = @elapsed Weights, Z, N, M, q = ReadFasta(filename, max_gap_fraction, theta, remove_dups)
#     println("preprocessing took $time seconds")
#     attention(Z, Weights; kwds...)
# end

# function minimize_pl(alg::PlmAlg, var::AttPlmVar;
#     initx0::Union{Nothing, Vector{Float64}} = nothing)
#     @extract var : N H d q2 LL = 2*H*N*d + H*q2
    
#     x0 = if initx0 === nothing 
#         rand(Float64, LL)
#     else 
#         initx0
#     end
#     pl = 0.0
#     parameters = zeros(LL)

#     opt = Opt(alg.method, length(x0))
#     ftol_abs!(opt, alg.epsconv)
#     xtol_rel!(opt, alg.epsconv)
#     xtol_abs!(opt, alg.epsconv)
#     ftol_rel!(opt, alg.epsconv)
#     maxeval!(opt, alg.maxit)
#     min_objective!(opt, (x, g) -> optimfunwrapper(g,x, var))
#     elapstime = @elapsed  (minf, minx, ret) = optimize(opt, x0)
#     numevals = opt.numevals
#     alg.verbose && @printf("pl = %.4f\t time = %.4f\t ", minf, elapstime)
#     alg.verbose && println("exit status = $ret")
#     pl = minf
#     parameters .= minx

#     return parameters, pl, elapstime, numevals
# end

# function pl_and_grad!(grad::Vector{Float64}, x::Vector{Float64}, plmvar::AttPlmVar)
    
#     @extract plmvar : H N M d q Z λ = N*q*lambda/M weights = W delta wdelta 
    
#     L = 2*H*N*d + H*q*q 
#     L == length(x) || error("Wrong dimension of parameter vector")
#     L == length(grad) || error("Wrong dimension of gradient vector")

#     Q = reshape(x[1:H*N*d],N,d,H)
#     K = reshape(x[H*N*d+1 : 2*H*N*d],N,d,H)
#     V = reshape(x[2*H*N*d+1:end],q,q,H)

#     pseudologlikelihood = zeros(Float64, N)
#     reg = zeros(Float64, N)

#     data = AttComputationQuantities(N,H,q)

     
#     Threads.@threads for site in 1:N 
#         pseudologlikelihood[site], reg[site] = update_Q_site!(grad, Z, Q[site,:,:], K, V, site, weights, λ, data, delta[site,:,:], wdelta)
#     end

#     Threads.@threads for site in 1:N 
#         update_K_site!(grad, Q, V, site, λ, data.sf, data.J, data.fact)
#     end

#     update_V!(grad, Q, V, λ, data) 
    
#     regularisation = sum(reg)
#     total_pslikelihood = sum(pseudologlikelihood) + regularisation
    
#     println(total_pslikelihood," ",regularisation)
#     return total_pslikelihood

# end

# function update_Q_site!(grad::Vector{Float64}, Z::Array{Int,2}, Q::Array{Float64,2}, K::Array{Float64,3}, V::Array{Float64,3}, site::Int, weights::Vector{Float64}, lambda::Float64, data::AttComputationQuantities, delta, wdelta)

#     # size(Q) == size(K) || error("Wrong dimensionality for Q and K")
#     d,H = size(Q)
#     q,q,H = size(V)
    
#     N,M = size(Z)

#     # print("1:") 
#     # @time 
#     @tullio W[j,h] := Q[d,h]*K[j,d,h] 
#     sf = softmax(W,dims=1)  #sf : N d (N_site)
    
    
#     # print("2:")
#     # @time 
#     @tullio J[j,a,b] := sf[j,h]*V[a,b,h]*(site!=j) #order HNq^2
#     # @tullio data.J[$site,j,a,b] = sf[h,j]*V[h,a,b]*(site!=j) #order HNq^2
#     # data.J[site,:,:,:] .= J_site 

#     # print("3:")
#     # @time 
#     @tullio mat_ene[m,a] := J[j,a,Z[j,m]]
#     # @tullio mat_ene[a,m] := data.J[$site,j,a,Z[j,m]] #order NMq
#     partition = sumexp(mat_ene,dims=2) #partition function for each m ∈ 1:M 

#     # print("4:")
#     # @tullio prob[a,m] := exp(mat_ene[a,m])/partition[m] #order Mq
#     # @time
#     @tullio probnew[m,a] := delta[m,a] - exp(mat_ene[m,a])/partition[m]
#     # @tullio probnew[a,m] := delta[$site,m,a] - exp(mat_ene[a,m])/partition[m]
#     lge = log.(partition) 

#     Z_site = view(Z,site,:)
    
#     # print("5:") 
#     # @time 
#     @tullio pl = weights[m]*(mat_ene[m,Z_site[m]] - lge[m]) #order M
#     pl *= -1

#     # print("6:")
#     # @tullio data.mat[$site,a,b,j] = weights[m]*(Z[j,m]==b)*((Z_site[m]==a)-prob[a,m]) (a in 1:q, b in 1:q) #order NMq^2 ORIGINAL
#     # @time
#     @tullio mat[j,a,b] := wdelta[j,m,a]*probnew[m,a] (a in 1:q, b in 1:q) 
#     # @time @tullio data.mat[$site,a,b,j] = wdelta[j,m,a]*probnew[a,m] (a in 1:q, b in 1:q) #order NMq^2 MIGLIORE?
#     # @tullio data.mat[j,a,b,$site] = wdelta[j,m,a]*probnew[a,m] (a in 1:q, b in 1:q)
#     mat[site,:,:] .= 0.0 
    

#     # print("7:")
#     # @time 
#     @tullio fact[j,h] := mat[j,a,b]*V[a,b,h]
#     # @tullio data.fact[$site,h,j] = data.mat[$site,a,b,j]*V[h,a,b] #order HNq^2
#     # @tullio data.fact[j,h,$site] = data.mat[a,b,j,$site]*V[h,a,b] #order HNq^2

    
#     outersum = zeros(Float64, N)

#     # print("8:")
#     # @time 
#     # J_ = permutedims(data.J,(2,3,4,1))
#     # V_ = permutedims(V,(2,3,1))
#     # K_ = permutedims(K,(3,2,1))
#     # sf_ = permutedims(sf,(2,1))
#     # fact_ = permutedims(data.fact,(3,2,1))
#     # @time 
#     @inbounds for counter in (site-1)*H*d + 1 : site*H*d 
#         _,y,h = counter_to_index(counter, N, d, q, H)
#         # print("1: ")
#         # @tullio innersum = K[$h,$y,j]*sf[$h,j] #order N
#         @tullio innersum = K[j,$y,$h]*sf[j,$h]

#         # print("2: ")
#         # @tullio outersum[j] = (K[$h,$y,j]*sf[$h,j] - sf[$h,j]*innersum) #order N
#         @tullio outersum[j] = (K[j,$y,$h]*sf[j,$h] - sf[j,$h]*innersum)
        
#         # print("3: ")
#         # @tullio scra = data.fact[$site,$h,j]*outersum[j] #order N
#         @tullio scra = fact[j,$h]*outersum[j]

#         # print("4: ")
#         # @tullio ∇reg =  data.J[$site,j,a,b]*V[$h,a,b]*outersum[j] #order Nq^2
#         @tullio ∇reg =  J[j,a,b]*V[a,b,$h]*outersum[j]
#         grad[counter] = -scra + 2*lambda*∇reg 
#     end
#     reg = lambda*L2Tensor(J) 
    

#     data.sf[site,:,:] .= sf
#     data.J[site,:,:,:] .= J 
#     data.mat[site,:,:,:] .= mat
#     data.fact[site,:,:] .= fact 

#     return pl, reg
# end

# function update_K_site!(grad::Vector{Float64}, Q::Array{Float64,3}, V::Array{Float64,3}, site::Int, lambda::Float64, sf::Array{Float64,3}, J::Array{Float64,4}, fact::Array{Float64,3}) 
#     N,d,H = size(Q)
#     _,q,_ = size(V)
    
#     scra = zeros(Float64,N,N)
#     scra1 = zeros(Float64,q,q)



#     # print("1:")
#     # @time 
#     @inbounds for counter in H*N*d+(site-1)*H*d+1:H*N*d + site*H*d
#         x,y,h = counter_to_index(counter, N, d, q, H) #h, lower dim, position
#         # print("1: ")
#         # @time 
#         @tullio scra[i,j] = Q[i, $y, $h]*(sf[i,j,$h]*(x==j) - sf[i,j,$h]*sf[i,$x,$h]) #order N^2
        
#         # print("2: ")
#         # @time 
#         @tullio scra2 = scra[i,j]*fact[i,j,$h] #order N^2
        
#         # print("3: ")
#         # @time 
#         @tullio scra1[a,b] = scra[i,j]*J[i,j,a,b] #order N^2q^2
        
#         # print("4: ")
#         # @time 
#         @tullio ∇reg = scra1[a,b]*V[a,b,$h] #order q^2
#         grad[counter] = - scra2 + 2*lambda*∇reg
#     end
#     return
# end

# function update_V!(grad::Vector{Float64}, Q::Array{Float64,3}, V::Array{Float64,3}, lambda::Float64, data)
    
#     N,d,H = size(Q)
#     q,q,H = size(V)

     

#     grad[2*N*d*H+1:2*N*H*d + H*q*q] .= 0.0
#     scra = zeros(Float64, N)
#     @inbounds for counter in 2*N*d*H+1:2*N*H*d + H*q*q
#         # print("1: ")
#         # @time 
#         a,b,h = counter_to_index(counter, N,d,q, H)
        
#         # print("2: ")
#         # @time 
#         @tullio scra[site] = data.mat[site,j,$a,$b]*data.sf[site,j,$h] (site in 1:N) #order N^2
#         # println("QUO")

#         # print("3: ")
#         # @time 
#         @tullio ∇reg = data.J[i,j,$a,$b]*data.sf[i,j,$h] #order N^2
        
#         grad[counter] = -sum(scra) + 2*lambda*∇reg
        
#     end
#     return
# end
# ## DOVREI CONTROLLARE counter_to_index




# ##############



# struct PlmAlg
#     method::Symbol
#     verbose::Bool
#     epsconv::Float64
#     maxit::Int
# end

# # struct PlmOut
# #     pslike::Union{Vector{Float64},Float64}
# #     Wtensor::Array{Float64,3}
# #     Vtensor::Union{Array{Float64,3},Array{Float64,4}}
# #     score::Array{Tuple{Int, Int, Float64},1}  
# # end



# struct AttPlmVar
#     N::Int
#     M::Int
#     d::Int
#     q::Int  
#     q2::Int
#     H::Int
#     lambda::Float64
#     Z::Array{Int,2} #MSA
#     W::Array{Float64,1} #weigths
#     delta::Array{Int,3}
#     # idx::Dict{Any,Any}
#     wdelta::Array{Float64,3}
#     function AttPlmVar(N,M,d,q,H,lambda,Z,Weigths)
#         idx = Dict()
#         @tullio delta[j,m,a] := Int(Z[j,m]==a) (a in 1:q)
#         @tullio wdelta[j,m,a] := Weigths[m]*delta[j,m,a]
#         # for a in 1:q
#         #     for j in 1:N
#         #         push!(idx, [j,a] => findall(x->x==a,Z[j,:]))
#         #     end
#         # end
        
#         new(N,M,d,q,q*q,H,lambda,Z,Weigths,delta,wdelta)
#     end
# end

# function Base.show(io::IO, AttPlmVar::AttPlmVar)
#     @extract AttPlmVar: N M d q H lambda
#     print(io,"AttPlmVar: \nN=$N\nM=$M\nq=$q\nH=$H\nd=$d\nλ=$(lambda)")
# end


# struct AttPlmOut
#     Q::Array{Float64,3}
#     K::Array{Float64,3}
#     V::Array{Float64,3}
#     pslike::Union{Vector{Float64},Float64}
# end

# function Base.show(io::IO, AttPlmOut::AttPlmOut)
#     @extract AttPlmOut: Q K V pslike
#     N,d,H = size(Q)
#     q,q,H = size(V) 
#     print(io,"AttPlmOut: \nsize(Q)=[$H,$d,$N]\nsize(K)=[$H,$d,$N]\nsize(V)=[$H,$q,$q]\npslike=$(pslike)")
# end


# struct AttComputationQuantities 
#     sf::Array{Float64,3}
#     J::Array{Float64,4}
#     mat::Array{Float64,4}
#     fact::Array{Float64,3}
#     function AttComputationQuantities(N,H,q)
#         sf = zeros(Float64, N, N, H)
#         J = zeros(Float64, N, N, q, q)
#         mat = zeros(Float64, N, N, q, q)
#         fact = zeros(Float64, N, N, H)
#         new(sf,J,mat,fact)
#     end
# end

        

