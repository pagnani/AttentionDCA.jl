function site_loglike(Z::Matrix,W,V,site)
    M = size(Z,2)
    Wsf_site = softmax(W[:,:,site],dims=2)
    @tullio J[a,b,j] := Wsf_site[h,j]*V[h,a,b]*(site!=j)
    
    @tullio mat_ene[a,m] := J[a,Z[j,m],j]

    pseudologlikelihood = 0.0 
    lge = logsumexp(mat_ene,dims=1)
    Z_site = view(Z,site,:)
    @tullio pseudologlikelihood = mat_ene[Z_site[m],m] - lge[m]
    pseudologlikelihood /= -M
    
    return pseudologlikelihood
end



function total_loglike(Z::Matrix{T},W,V,lambda,weights) where T<:Integer
    M = size(Z,2)
    sumweights = sum(weights)
    Wsf_site = softmax(W,dims=2)
    @tullio J[a,b,i,r] := Wsf_site[h,i,r]*V[h,a,b]*(i!=r)
    
    @tullio mat_ene[a,r,m] := J[a,Z[j,m],j,r]
    
    lge = logsumexp(mat_ene,dims=1)[1,:,:]
    @tullio pseudologlikelihood = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pseudologlikelihood /= -sumweights
    
    pseudologlikelihood += lambda*L2Tensor(W) + lambda*L2Tensor(V)

    return pseudologlikelihood
end

function total_loglikeJreg(Z::Matrix{T},W,V,lambda,weights;q=21) where T<:Integer
    N,M = size(Z)
    
    Wsf_site = softmax(W,dims=2)
    @tullio J[a,b,i,r] := Wsf_site[h,i,r]*V[h,a,b]*(i!=r)
    
    @tullio mat_ene[a,r,m] := J[a,Z[j,m],j,r]
    
    lge = logsumexp(mat_ene,dims=1)[1,:,:]
    @tullio pseudologlikelihood = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pseudologlikelihood /= -1
    
    pseudologlikelihood += N*q*lambda*(L2Tensor(J))/M 

    return pseudologlikelihood
end


function parallel_total_loglike(Z::Matrix{T},W,V,lambda,weights) where T<:Integer
    M = size(Z,2)
    Wsf_site = softmax(W,dims=2)
    @tullio J[a,b,i,r] := Wsf_site[h,i,r]*V[h,a,b,r]*(i!=r)
    
    @tullio mat_ene[a,r,m] := J[a,Z[j,m],j,r]
    
    lge = logsumexp(mat_ene,dims=1)[1,:,:]
    @tullio pseudologlikelihood = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pseudologlikelihood /= -1
    
    pseudologlikelihood += lambda*L2Tensor(W) + lambda*L2Tensor(V)

    return pseudologlikelihood
end


function fa_total_pslikelihood(Z, Q, K, V, weights, lambda)
    M = length(weights)
    H,d,N = size(Q)
    _,q,_ = size(V)
    λ = q*N*lambda/M

    @tullio W[h,i,j] := Q[h,d,i]*K[h,d,j]
    sf = softmax_notinplace(W,dims=3)
    @tullio J[a,b,i,r] := sf[h,r,i]*V[h,a,b]*(i!=r)

    @tullio mat_ene[a,r,m] := J[a,Z[j,m],j,r]
    
    lge = logsumexp(mat_ene,dims=1)[1,:,:]
    @tullio pseudologlikelihood = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pseudologlikelihood *= -1

    reg = λ*L2Tensor(J)
    pseudologlikelihood = pseudologlikelihood + reg
    println(pseudologlikelihood," ",reg)
    return pseudologlikelihood
   
end



function softmax_notinplace(x::AbstractArray; dims = 1)
    max_ = maximum(x; dims)
    if all(isfinite, max_)
        @fastmath out = exp.(x .- max_)
    else
        @fastmath @. out = ifelse(isequal(max_,Inf), ifelse(isequal(x,Inf), 1, 0), exp(x - max_))
    end
    return out ./ sum(out; dims)
end



function ar_likelihood(Z,Q,K,V,lambda,weights)
    H,d = size(Q)
    H,q,_ = size(V)
    N,M = size(Z)
    
    λ = N*q*lambda/M
    
    mask = zeros(N,N,H)
    # mask = fill(-10000, N,N,H)
    # for i in 1:H
    #     mask[:,:,i] = UpperTriangular(mask[:,:,i])
    # end
    @tullio mask[i,j,h] := -10000*(j>=i) (i in 1:N, j in 1:N, h in 1:H)

    @tullio scra_sf[i, j, h] := Q[h,d,i]*K[h,d,j] + mask[i,j,h]
    # sf = scra_sf + mask 
    sf = softmax_notinplace(scra_sf./sqrt(d),dims=2) 

    # @tullio J[i,j,a,b] := sf[i,j,h]*V[h,a,b]*(j<i)
    @tullio J[i,j,a,b] := sf[i,j,h]*V[h,a,b]*(i!=1)

    @tullio mat_ene[a,r,m] := J[r,j,a,Z[j,m]]
    lge = logsumexp(mat_ene,dims=1)[1,:,:]

    @tullio pl = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pl = -1*pl
    reg = λ*L2Tensor(J)
    
    pl = pl + reg
    return pl
end 

function likelihood(Z,Q,K,V,lambda,weights)
    H,d = size(Q)
    H,q,_ = size(V)
    N,M = size(Z)
    
    λ = N*q*lambda/M

    @tullio sf[i, j, h] := Q[h,d,i]*K[h,d,j]
    sf = softmax_notinplace(sf,dims=2) 

    @tullio J[i,j,a,b] := sf[i,j,h]*V[h,a,b]*(j!=i)

    @tullio mat_ene[a,r,m] := J[r,j,a,Z[j,m]]
    lge = logsumexp(mat_ene,dims=1)[1,:,:]

    @tullio pl = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pl = -1*pl
    reg = λ*L2Tensor(J)
    
    pl = pl + reg
    return pl
end 