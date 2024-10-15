using AttentionDCA, Statistics, PyPlot, LinearAlgebra, Tullio, JLD2


#Functions to get the k largest elements from an array of unspecified dimensionality 

function one_hot(msa::Array{Int,2})
    N, M = size(msa)
    new_msa = zeros(N*21,M)
    for i in 1:N
        for j in 1:M
            index = msa[i,j]  
            new_msa[(i-1)*21 + index, j] = 1
        end
    end
    return new_msa
end

function one_hot(seq::Array{Int,1})
    N = length(seq)
    new_msa = zeros(N*21)
    for j in 1:N
        index = seq[j]  
        new_msa[(j-1)*21 + index] = 1
    end
    return Matrix{}(new_msa')
end


function getci!(M,k,cmins,vmins) # inplace
    ci = CartesianIndices(size(M))
    for i in 1:k
        cmins[i] = ci[i]
        vmins[i] = M[i]
    end
    imin = findmin(vmins)[2]
    for i in firstindex(M)+k:lastindex(M)
        if M[i] > vmins[imin]
            cmins[imin] = ci[i]
            vmins[imin] = M[i]
            imin = findmin(vmins)[2]
        end
    end
    return cmins, vmins
end


function getci(M,k) # allocating
    cmins = Vector{CartesianIndex{length(size(M))}}(undef,k)
    vmins = Vector{eltype(M)}(undef,k)
    return getci!(M,k,cmins,vmins)
end




function true_structure(structfile; min_separation=0, cutoff=8.0)
    dist = AttentionDCA.compute_residue_pair_dist(structfile)
    for (key,value) in dist
            if key[2]-key[1]<=min_separation || value > cutoff || value == 0 
                delete!(dist,key)
            end
    end
    dist = reduce(hcat,getindex.(collect(keys(dist)),i) for i in 1:2)
    return dist
end

#Given a LxL matrix, representing the positional information of the alignment, this function computes its coupling score
score_from_matrix(A) = sort([(j,i,A[j,i]) for i in 2:size(A,2) for j in 1:i-1], by = x->x[3], rev = true)


#Given Q,K,V this function computes the attention matrices for each head, if sym = true the matrices are symmetrised 
function attention_heads(Q,K,V; sym =false)
    
    @tullio W[h,i,j] := Q[h,d,i]*K[h,d,j]
    A = AttentionDCA.softmax(W, dims=3)
    @tullio A[h,i,j] *= (i!=j)
    
    
    if sym 
        for h in 1:size(A,1)
            A[h,:,:] = (A[h,:,:] + A[h,:,:]')/2
        end
    end
    
    return A
end

#Given the attention matrices, this function returns an LxL matrix with k highest elements for each head by either averaging (foo = mean) them or by taking the maximimum (foo = maximum) for each (i,j) couple.
function k_matrix(Q,K,V, k, version; sym = true, APC = false, sqr = false)
    
    version in [mean, maximum] || error("mean and maximum only supported versions")
    
    
    A = attention_heads(Q,K,V, sym = sym)
    H,N,N = size(A)
    M = zeros(N,N)
    
    if sqr
        A = A.*A
    end
    
    if k >= N*(N-1)/2 
        M = version(A, dims = 1)[1,:,:]
        M = (M + M')/2
        if APC 
            M = AttentionDCA.correct_APC(M)
        end
        
        return M, A
    end
    
    _A = zeros(H,N,N)
    cmins = Vector{CartesianIndex{2}}(undef,k)
    vmins = Vector{eltype(A)}(undef,k)
    for h in 1:H
        getci!(A[h,:,:], k, cmins, vmins)
        _A[h,cmins].=vmins
    end
    
    
    
    if version == maximum
        M = maximum(_A, dims = 1)[1,:,:]
    end
    
    if version == mean
        for i in 1:N
            for j in 1:N
                n = sum((_A[:,i,j].!=0.0))
                if n > 0
                    M[i,j] = sum(_A[:,i,j])/n
                end
            end
        end
    end
     
    M = (M + M')/2

    
    if APC 
        M = AttentionDCA.correct_APC(M)
    end
    
    return M, _A
end


k_matrix(Q,K,V, version; sym = true, APC = false, sqr = false) = k_matrix(Q,K,V, size(Q,3)^2,version; sym = sym, APC = APC, sqr = sqr)

k_score(Q,K,V, k, version; sym = true, APC = false, sqr = false) = score_from_matrix(k_matrix(Q,K,V, k, version, sym = sym, APC = APC, sqr = sqr)[1])

k_score(M) = score_from_matrix(M)

attention_PPV(M, structure) = compute_PPV(k_score(M), structure)

attention_PPV(Q,K,V, k, version, structure; sym = true, APC = false, sqr = false) = compute_PPV(k_score(Q,K,V, k, version, sym = sym, APC = APC, sqr = sqr), structure)

attention_PPV(Q,K,V, version, structure; sym = true, APC = false, sqr = false) = compute_PPV(k_score(Q,K,V, size(Q,3)^2, version, sym = sym, APC = APC, sqr = sqr), structure)

#Given a LxL matrix, representing the positional information of the alignment, this function computes its coupling score
score_from_matrix(A) = sort([(j,i,A[j,i]) for i in 2:size(A,2) for j in 1:i-1], by = x->x[3], rev = true)
