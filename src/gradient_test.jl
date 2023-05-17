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



function ar2_likelihood(Z,Q,K,V,lambda,weights)
    H,d = size(Q)
    H,q,_ = size(V)
    N,M = size(Z)
    
    numpar = N*(N-1)*q*q
    # numpar = 1.0
    λ = lambda/numpar
    
    mask = zeros(N,N,H)
    
    @tullio mask[i,j,h] := -10000*(j>=i) (i in 1:N, j in 1:N, h in 1:H)

    @tullio scra_sf[i, j, h] := Q[h,d,i]*K[h,d,j] + mask[i,j,h] 
    sf = softmax_notinplace(scra_sf./sqrt(d),dims=2) 

    @tullio J[i,j,a,b] := sf[i,j,h]*V[h,a,b]*(i!=1)

    @tullio mat_ene[a,r,m] := J[r,j,a,Z[j,m]]
    lge = logsumexp(mat_ene,dims=1)[1,:,:]

    @tullio pl = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pl = -1*pl
    reg = λ*L2Tensor(J)
    
    pl = pl + reg
    # return pl, J, mat_ene, lge, pl - reg, reg
    return pl
end 

function ar_likelihood(Z,Q,K,V,lambda,weights)
    H,d = size(Q)
    H,q,_ = size(V)
    N,M = size(Z)
    
    numpar = N*(N-1)*q*q
    # numpar = 1.0
    λ = lambda/numpar
    
    @tullio scra_sf[i, j, h] := Q[h,d,i]*K[h,d,j] 
    sf = softmax_notinplace(scra_sf./sqrt(d),dims=2) 

    @tullio J[i,j,a,b] := sf[i,j,h]*V[h,a,b]*(j<i)

    @tullio mat_ene[a,r,m] := J[r,j,a,Z[j,m]]
    lge = logsumexp(mat_ene,dims=1)[1,:,:]

    @tullio pl = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pl = -1*pl
    reg = λ*L2Tensor(J)
    pl = pl + reg
    # return pl, J, mat_ene, lge, pl - reg, reg
    println(pl," ",reg)
    return pl
    # return sf, J, mat_ene  
end

function field_ar_likelihood(Z,Q,K,V,F,lambdaJ,lambdaF,weights; dd = size(Q,2))
    H,d = size(Q)
    H,q,_ = size(V)
    N,M = size(Z)
    
    numpar = N*(N-1)*q*q
    # numpar = 1.0
    λJ = lambdaJ
    λF = lambdaF

    @tullio scra_sf[i, j, h] := Q[h,d,i]*K[h,d,j] 
    sf = softmax_notinplace(scra_sf./sqrt(dd),dims=2) 

    @tullio J[i,j,a,b] := sf[i,j,h]*V[h,a,b]*(j<i)

    @tullio _mat_ene[a,r,m] := J[r,j,a,Z[j,m]] 
    @tullio mat_ene[a,r,m] := _mat_ene[a,r,m] + F[a,r]
    lge = logsumexp(mat_ene,dims=1)[1,:,:]

    @tullio pl = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pl = -1*pl
    reg = λJ*L2Tensor(J) + λF*L2Tensor(F) 
    pl = pl + reg
    # return pl, J, mat_ene, lge, pl - reg, reg
    println(pl," ",reg)
    return pl
    # return sf, J, mat_ene  
end

function likelihood(Z,Q,K,V,lambda,weights; dd = size(Q,2))
    H,d = size(Q)
    H,q,_ = size(V)
    N,M = size(Z)
    numpar = N*(N-1)*q*q
    # numpar = 1.0
    # λ = lambda/numpar
    λ = lambda

    @tullio sf[i, j, h] := Q[h,d,i]*K[h,d,j]
    sf = softmax_notinplace(sf./sqrt(dd),dims=2) 
    
    @tullio J[i,j,a,b] := sf[i,j,h]*V[h,a,b]*(j!=i)
   
    @tullio mat_ene[a,r,m] := J[r,j,a,Z[j,m]]
    lge = logsumexp(mat_ene,dims=1)[1,:,:]

    @tullio pl = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pl = -1*pl
    reg = λ*sum(abs2,J)
    # println(reg)
    pl = pl + reg
    #println(pl," ",reg)
    # return pl, J, mat_ene, lge, pl-reg, reg
   return pl
end 


function fieldlikelihood(Z,Q,K,V,F,lambdaJ, lambdaF, weights; dd = size(Q,2))
    H,d = size(Q)
    H,q,_ = size(V)
    N,M = size(Z)
    numpar = N*(N-1)*q*q
    # numpar = 1.0
    # λJ = lambdaJ/numpar
    # λF = lambdaF/(N*q)

    λJ = lambdaJ
    λF = lambdaF

    @tullio sf[i, j, h] := Q[h,d,i]*K[h,d,j]
    sf = softmax_notinplace(sf./sqrt(dd),dims=2) 
    
    @tullio J[i,j,a,b] := sf[i,j,h]*V[h,a,b]*(j!=i)
   
    @tullio _mat_ene[a,r,m] := J[r,j,a,Z[j,m]] 
    @tullio mat_ene[a,r,m] := _mat_ene[a,r,m] + F[a,r]
    lge = logsumexp(mat_ene,dims=1)[1,:,:]

    @tullio pl = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pl = -1*pl
    reg = λJ*L2Tensor(J) + λF*L2Tensor(F) 
    pl = pl + reg
    println(pl," ",reg)
   return pl
end

function likelihood(Z,Q,K,V,λQ,λV,weights; dd = size(Q,2))
    H,d = size(Q)
    H,q,_ = size(V)
    N,M = size(Z)
    numpar = N*(N-1)*q*q
    # numpar = 1.0
    # λ = lambda/numpar
    

    @tullio sf[i, j, h] := Q[h,d,i]*K[h,d,j]
    sf = softmax_notinplace(sf./sqrt(dd),dims=2) 
    
    @tullio J[i,j,a,b] := sf[i,j,h]*V[h,a,b]*(j!=i)
   
    @tullio mat_ene[a,r,m] := J[r,j,a,Z[j,m]]
    lge = logsumexp(mat_ene,dims=1)[1,:,:]

    @tullio pl = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pl = -1*pl
    reg = λQ*L2Tensor(Q) + λQ*L2Tensor(K) + λV*L2Tensor(V)
    # println(reg)
    pl = pl + reg
    println(pl," ",reg)
    # return pl, J, mat_ene, lge, pl-reg, reg
   return pl
end 

function fieldlikelihood(Z,Q,K,V,F,lambdaQ, lambdaV, lambdaF, weights; dd = size(Q,2))
    H,d = size(Q)
    H,q,_ = size(V)
    N,M = size(Z)
    
    # numpar = 1.0
    # λJ = lambdaJ/numpar
    # λF = lambdaF/(N*q)

    λQ = lambdaQ
    λV = lambdaV
    λF = lambdaF

    @tullio sf[i, j, h] := Q[h,d,i]*K[h,d,j]
    sf = softmax_notinplace(sf./sqrt(dd),dims=2) 
    
    @tullio J[i,j,a,b] := sf[i,j,h]*V[h,a,b]*(j!=i)
   
    @tullio _mat_ene[a,r,m] := J[r,j,a,Z[j,m]] 
    @tullio mat_ene[a,r,m] := _mat_ene[a,r,m] + F[a,r]
    lge = logsumexp(mat_ene,dims=1)[1,:,:]

    @tullio pl = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pl = -1*pl
    reg = λQ*L2Tensor(Q) + λQ*L2Tensor(K) + λV*L2Tensor(V) + λF*L2Tensor(F) 
    pl = pl + reg
    println(pl," ",reg)
   return pl
end