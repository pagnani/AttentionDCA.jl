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

function total_loglikeJreg(Z::Matrix{T},W,V,lambda,weights) where T<:Integer
    M = size(Z,2)
    sumweights = sum(weights)
    Wsf_site = softmax(W,dims=2)
    @tullio J[a,b,i,r] := Wsf_site[h,i,r]*V[h,a,b]*(i!=r)
    
    @tullio mat_ene[a,r,m] := J[a,Z[j,m],j,r]
    
    lge = logsumexp(mat_ene,dims=1)[1,:,:]
    @tullio pseudologlikelihood = weights[m]*(mat_ene[Z[r,m],r,m] - lge[r,m])
    pseudologlikelihood /= -sumweights
    
    pseudologlikelihood += lambda*L2Tensor(J) 

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