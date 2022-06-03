# function pslikelihood(x,var::PlmVar,Z,weights,limits)
#     N = var.N
#     M = var.M
#     H = var.H
#     q = var.q
#     λ = N*q*(var.lambda)/M

#     L = H*N*N + H*q*q #total numbero of parameters
#     L == length(x) || error("Wrong dimension of parameter vector")
#     #L == length(grad) || error("Wrong dimension of gradient vector")

#     W = reshape(x[1:H*N*N],H,N,N)
#     V = reshape(x[H*N*N+1:end],H,q,q)

#     pseudologlikelihood = zeros(Float64, N)
#     l2 = zeros(Float64, N)
#     grad = zeros(Float64, L)
     
#     J = zeros(Float64,q,q,N,N)
#     Wsf = zeros(Float64,H,N,N)
#     mat = zeros(Float64,N,N,q,q)


#     Threads.@threads for site = 1:N
#         pseudologlikelihood[site],l2[site],J[:,:,:,site],Wsf[:,:,site],mat[site,:,:,:] = new_update_gradW_site_Jreg!(grad,Z,W,V,weights,λ,site,limits)
#     end
    
    
#     new_update_gradV_Jreg!(grad,Z,Wsf,V,λ,J,mat)

   
#     # pg == pointer(G) || error("Different pointer")
    
#     total_loglike = sum(pseudologlikelihood)
#     L2 = sum(l2)
#     # if dist !== nothing 
#     #     score = compute_dcascore(W,V)
#     #     roc = compute_referencescore(score, dist)
#     #     rocN = roc[N][end]
#     #     rocN5 = roc[div(N,5)][end]
        
#     #     file !== nothing && write(file, "$total_loglike   "*"$L2   "*"$rocN5   "*"$rocN"*"\n")
#     #     verbose && println("$total_loglike   "*"$L2   "*"$rocN5   "*"$rocN")
#     # else
#     #     file !== nothing && write(file, "$total_loglike   "*"$L2"*"\n")
#     #     verbose && println("$total_loglike   "*"$L2 ")
#     # end
#     println(total_loglike+L2,"  ", L2)
#     return total_loglike + L2, grad 
    
# end



# function new_update_gradW_site_Jreg!(grad,Z,W,V,weights,λ,site,limits)
#     pg = pointer(grad)
#     N,_ = size(Z)
#     H,q,_ = size(V)

#     W_site = view(W,:,:,site)
#     Wsf_site = softmax(W_site,dims=2)
    
    
#     @tullio J[a,b,j] := Wsf_site[h,j]*V[h,a,b]*(site!=j)
#     @tullio mat_ene[a,m] := J[a,Z[j,m],j]
    
    
#     pl = 0.0
#     partition = sumexp(mat_ene,dims=1)
#     @tullio prob[a,m] := exp(mat_ene[a,m])/partition[m]
    
#     lge = log.(partition)
#     Z_site = view(Z,site,:)
#     @tullio pl = weights[m]*(mat_ene[Z_site[m],m] - lge[m])
#     pl *= -1
    

#     grad[(site-1)*N*H+1:site*N*H] .= 0.0

#     weights_minibatch = weights[limits]
#     Z_minibatch = view(Z,:,limits)
#     Z_site_minibatch = view(Z_site,limits)
#     prob_minibatch = view(prob,:,limits)

#     @tullio mat[j,a,b] := weights_minibatch[m]*(Z_minibatch[j,m]==b)*((Z_site_minibatch[m]==a)-prob_minibatch[a,m]) (a in 1:q, b in 1:q)
#     mat[site,:,:] .= 0.0
#     @tullio fact[j,h] := mat[j,a,b]*V[h,a,b]
#     @tullio fact2[j,h] := 2*J[a,b,j]*V[h,a,b]

#     @inbounds for counter in (site-1)*N*H+1:site*N*H 
#         gradL2 = 0.0
#         h,i,r = counter_to_index(counter,N,q,H)
#         if r == site 
#             @simd for j = 1:N 
#                 scra = ((i==j)*Wsf_site[h,i]-Wsf_site[h,i]*Wsf_site[h,j])
#                 grad[counter] += fact[j,h] *scra
#                 gradL2 += fact2[j,h]*scra
#             end
#         end
#         grad[counter] *= -1
#         grad[counter] += λ*gradL2

#     end

#     pg == pointer(grad) || error("Different pointer")
#     return pl, λ*L2Tensor(J), J, Wsf_site , mat
# end


# function new_update_gradV_Jreg!(grad,Z,Wsf,V,λ,J,mat)
#     pg = pointer(grad)

#     N,_ = size(Z)
#     H,q,_ = size(V)

#     L = H*N*N + H*q*q

#     grad[H*N*N+1:end] .= 0.0

#     @inbounds for counter = H*N*N+1:L 
#         L2 = 0.0
#         h,c,d = counter_to_index(counter,N,q,H)
#         Wsf_h = view(Wsf,h,:,:)
#         mat_cd = view(mat,:,:,c,d)
#         @tullio g = Wsf_h[j,site]*mat_cd[site,j]
#         grad[counter] = g
#         J_cd = view(J,c,d,:,:)
#         @tullio L2 = J_cd[y,x]*Wsf_h[y,x]        
        
#         grad[counter] *= -1     
#         grad[counter] += 2*λ*L2
#     end 

#     pg == pointer(grad) || error("Different pointer")

#     return 
# end



# function my_minimiser(opt, x, var, structure, Z, weights; maxit=1000, length_minibatches = 100)
#     N,M = size(Z)
#     H = var.H 
#     q = var.q
#     perm = shuffle(1:M)
#     Z = Z[:,perm]
#     weights = weights[perm]
#     number_minibatches = M ÷ length_minibatches
#     j = 0
#     finitial,grad = pslikelihood(x,var,Z, weights, 1:length_minibatches)
#     xinitial = copy(x)
#     f1 = 0
#     for i in 1:maxit 
#         j += 1 
#         if j != number_minibatches
#             update!(opt, x, grad)
#             f1,grad = pslikelihood(x,var,Z, weights, j*length_minibatches + 1 : (j+1)*length_minibatches)
        
#             finitial = copy(f1)
#             xinitial = copy(x)
#         else
#             update!(opt, x, grad)
#             f1,grad = pslikelihood(x,var,Z, weights,(number_minibatches-1)*length_minibatches+1:M)
        
            
#             finitial = copy(f1)
#             xinitial = copy(x)
#             j = 0
#         end
#     end

    


#     W = reshape(x[1:H*N*N],H,N,N)
#     V = reshape(x[H*N*N+1:end], H,q,q)
#     score = compute_dcascore(W,V)
#     dist = compute_residue_pair_dist(structure)
#     roc = map(x->x[4],compute_referencescore(score, dist))
#     return x, f1, "Maxeval reached", score, roc


# end


# function fa_pslikelihood(x, plmvar,limits, Z, weights)

    
#     H = plmvar.H 
#     N = plmvar.N
#     M = plmvar.M
#     d = plmvar.d
#     q = plmvar.q
#     λ = N*q*(plmvar.lambda)/M
    
    
#     L = 2*H*N*d + H*q*q 
#     L == length(x) || error("Wrong dimension of parameter vector")
    

#     grad = zeros(Float64, L)

#     Q = reshape(x[1:H*N*d],H,d,N)
#     K = reshape(x[H*N*d+1 : 2*H*N*d],H,d,N)
#     V = reshape(x[2*H*N*d+1:end],H,q,q)

#     pseudologlikelihood = zeros(Float64, N)
#     reg = zeros(Float64, N)

#     J = zeros(Float64, N,N,q,q)
#     sf = zeros(Float64, H, N, N)
#     mat = zeros(Float64, N, q, q, N) # [site, a,b,j]
#     fact = zeros(Float64, N, H, N) #[site, h,j]   
#     Threads.@threads for site in 1:N 
#         pseudologlikelihood[site], reg[site], sf[:,site,:], mat[site, :, :, :], fact[site, :, :], J[site,:,:,:] = new_update_Q_site!(grad, Z, Q, K, V, site, weights,λ)
#     end
    
#     Threads.@threads for site in 1:N 
#         update_K_site!(grad, Q, V, site, sf, fact, J, λ)
#     end

#     update_V!(grad, Q, V, mat, sf, J, λ)
    
#     regularisation = sum(reg)
#     total_pslikelihood = sum(pseudologlikelihood) + regularisation
    
    
    

#     println(total_pslikelihood," ",regularisation)
#     return total_pslikelihood, grad

# end


function new_fa_pl_and_grad!(grad, x, Z, weights, lambda, limits; H = 32, d = 10, q = 21)
    pg = pointer(grad)
    
    N,M = size(Z)
    λ = N*q*lambda/M
    
    L = 2*H*N*d + H*q*q 
    L == length(x) || error("Wrong dimension of parameter vector")
    L == length(grad) || error("Wrong dimension of gradient vector")

    Q = reshape(x[1:H*N*d],H,d,N)
    K = reshape(x[H*N*d+1 : 2*H*N*d],H,d,N)
    V = reshape(x[2*H*N*d+1:end],H,q,q)

    pseudologlikelihood = zeros(Float64, N)
    reg = zeros(Float64, N)

    data = FAComputationQuantities(N,H,q)

    
    Threads.@threads for site in 1:N 
        pseudologlikelihood[site], reg[site] = new_update_Q_site!(grad, Z, Q, K, V, site, weights,λ,data,limits)
    end
    
    Threads.@threads for site in 1:N 
        update_K_site!(grad, Q, V, site, λ, data.sf, data.J, data.fact)
    end

    update_V!(grad, Q, V, λ, data)
    
    regularisation = sum(reg)
    total_pslikelihood = sum(pseudologlikelihood) + regularisation
    
    
    
    pg == pointer(grad) || error("Different pointer")
    println(total_pslikelihood," ",regularisation)
    return total_pslikelihood

end









function fa_minimiser(opt, x, var, structure; maxit=1000, length_minibatches = 100)
    Z = var.Z
    weights = var.W 
    N,M = size(Z)

    H = var.H 
    q = var.q
    d = var.d
    lambda = var.lambda

    perm = shuffle(1:M)    
    Z = Z[:,perm]
    weights = weights[perm]
    number_minibatches = M ÷ length_minibatches
    j = 0
    grad = zeros(length(x))
    pl = new_fa_pl_and_grad!(grad, x, Z, weights, lambda, j*length_minibatches+1:(j+1)*length_minibatches, H = H, q = q, d = d) 
    for i in 1:maxit 
         j += 1 
         if j != number_minibatches
            update!(opt, x, grad)
            pl = new_fa_pl_and_grad!(grad, x, Z, weights, lambda, j*length_minibatches+1:(j+1)*length_minibatches, H = H, q = q, d = d)
        else
            update!(opt, x, grad)
            pl =  new_fa_pl_and_grad!(grad, x, Z, weights, lambda, (number_minibatches-1)*length_minibatches+1:M, H = H, q = q, d = d)
            j = 0
        end
    end

    


    Q = reshape(x[1:H*d*N],H,d,N)
    K = reshape(x[H*d*N+1:2*H*d*N],H,d,N)
    V = reshape(x[2*H*d*N+1:end], H,q,q)
    score = compute_dcascore_fa(Q,K,V)
    dist = compute_residue_pair_dist(structure)
    roc = map(x->x[4],compute_referencescore(score, dist))
    return x, pl, score, roc
    
end


function new_update_Q_site!(grad, Z, Q, K, V, site, weights,λ,data,limits)
    pg = pointer(grad)
    size(Q) == size(K) || error("Wrong dimensionality for Q and K")
    H,d,N = size(Q)
    H,q,_ = size(V)
    

    @tullio W[h, j] := Q[h,d,$site]*K[h,d,j]
    sf = softmax(W,dims=2)
    data.sf[:,site,:] = softmax(W,dims=2)

    @tullio J_site[j,a,b] := sf[h,j]*V[h,a,b]*(site!=j)
    data.J[site,:,:,:] = J_site

    @tullio mat_ene[a,m] := data.J[$site,j,a,Z[j,m]] 
    partition = sumexp(mat_ene,dims=1) #partition function for each m ∈ 1:M

    @tullio prob[a,m] := exp(mat_ene[a,m])/partition[m]
    lge = log.(partition)

    Z_site = view(Z,site,:)
    @tullio pl = weights[m]*(mat_ene[Z_site[m],m] - lge[m])
    pl *= -1

    Z = view(Z,:,limits)
    weights = view(weights,limits)
    Z_site = view(Z_site,limits)
    prob = view(prob,:,limits)

    @tullio mat[a,b,j] := weights[m]*(Z[j,m]==b)*((Z_site[m]==a)-prob[a,m]) (a in 1:q, b in 1:q)
    mat[:,:,site] .= 0.0
    data.mat[site,:,:,:] = mat

    @tullio fact[h,j] := mat[a,b,j]*V[h,a,b]
    data.fact[site,:,:] = fact

    for counter in (site-1)*H*d + 1 : site*H*d
        h,y,x = new_counter_to_index(counter, N, d, q, H)
        @tullio innersum = K[$h,$y,j]*sf[$h,j]
        @tullio outersum[j] := (K[$h,$y,j]*sf[$h,j] - sf[$h,j]*innersum) 
        @tullio scra := fact[$h,j]*outersum[j]
        @tullio ∇reg :=  J_site[j,a,b]*V[$h,a,b]*outersum[j]
        grad[counter] = -scra + 2*λ*∇reg 
    end
    
    pg == pointer(grad) || error("Different pointer")
    reg = λ*L2Tensor(J_site)
    return pl, reg

end

