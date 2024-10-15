function compute_residue_pair_dist(filedist::String)
    d = readdlm(filedist)
    if size(d,2) == 4 
        return Dict((round(Int,d[i,1]),round(Int,d[i,2])) => d[i,4] for i in 1:size(d,1) if d[i,4] != 0)
    elseif size(d,2) == 3
        Dict((round(Int,d[i,1]),round(Int,d[i,2])) => d[i,3] for i in 1:size(d,1) if d[i,3] != 0)
    end

end

function compute_referencescore(score,dist::Dict; mindist::Int=6, cutoff::Number=8.0)
    nc2 = length(score)
    out = Tuple{Int,Int,Float64,Float64}[]
    ctrtot = 0
    ctr = 0
    for i in 1:nc2
        sitei,sitej,plmscore = score[i][1],score[i][2], score[i][3]
        dij = if haskey(dist,(sitei,sitej)) 
            dist[(sitei,sitej)]
        else
           continue
        end
        if sitej - sitei >= mindist 
            ctrtot += 1
            if dij < cutoff
                ctr += 1
            end
            push!(out,(sitei,sitej, plmscore, ctr/ctrtot))
        end
    end 
    out
end



"""
    score(Q, K, V; min_separation::Int=6
Function to compute the Frobenious contact score of the interaction matrix given the (Q,K,V) matrices. \n 
Argument ‘min_separation’ is the minimum separation between the residues to be considered as a possible non-trivial contact. Default value is 6. \n
The function returns a vector of tuple containing the residue pair and their contact score.

# Examples
```
julia> out = trainer("file.fasta",20,n_epochs=100);
julia> s = score(out.m..., min_separation = 6)
```
"""
function score(Q, K, V; min_separation::Int=6)
    
    H,d,L = size(Q)
    H,d,L = size(K)
    H,q,q = size(V)
    @tullio W[h,i,j] := Q[h,d,i]*K[h,d,j] 
    W = softmax(W, dims=3)
    @tullio Jtens[a, b, r, i] := W[h, r, i] * V[h, a, b] * (i != r) (i in 1:L)

    Jt = 0.5 * (Jtens + permutedims(Jtens,[2,1,4,3]))

    ht = zeros(eltype(Jt), q, L)
    Jzsg, _ = gauge(Jt, ht, ZeroSumGauge())
    FN = compute_fn(Jzsg)
    FNapc = correct_APC(FN)
    return compute_ranking(FNapc, min_separation)
end

score(m::NamedTuple; min_separation::Int=6) = score(m..., min_separation = min_separation)
score(out::AttOutStd, kwds...) = score(out.m, kwds...)
score(out::AttOutAr, kwds...) = epistatic_score(out.ArNet, out.ArVar, 1, min_separation = 6, kwds...)



"""
    score(Jtens; min_separation::Int=6)
Function to compute the Frobenious contact score directly from the interaction matrix,
"""
function score(Jtens; min_separation::Int=6)
    q,q,L,L = size(Jtens)
    
    Jt = 0.5 * (Jtens + permutedims(Jtens,[2,1,4,3]))

    ht = zeros(eltype(Jt), q, L)
    Jzsg, _ = gauge(Jt, ht, ZeroSumGauge())
    FN = compute_fn(Jzsg)
    FNapc = correct_APC(FN)
    return compute_ranking(FNapc, min_separation)
end

function compute_fn(J::AbstractArray{T,4}) where {T<:AbstractFloat}
    q, q, L, L = size(J)
    fn = zeros(T, L, L)
    for i in 1:L
        # s = zero(T)
        for j in 1:L
            s = zero(T)
            for a in 1:q-1, b in 1:q-1
                s += J[a, b, i, j]^2
            end
            fn[i, j] = s
        end
    end
    # return fn
    return (fn + fn') * T(0.5)
end

function correct_APC(S::Matrix)
    N = size(S, 1)
    Si = sum(S, dims=1)
    Sj = sum(S, dims=2)
    Sa = sum(S) * (1 - 1 / N)
    S -= (Sj * Si) / Sa
    return S
end

function compute_ranking(S::Matrix{Float64}, min_separation::Int = 6)
    N = size(S, 1)
    R = Array{Tuple{Int,Int,Float64}}(undef, div((N-min_separation)*(N-min_separation+1), 2))
    counter = 0
    for i = 1:N-min_separation, j = i+min_separation:N
        counter += 1
        R[counter] = (i, j, S[j,i])
    end

    sort!(R, by=x->x[3], rev=true)
    return R 
end


"""
    compute_PPV(score::Vector{Tuple{Int,Int,Float64}}, filestruct::String; min_separation::Int = 6)
Function to compute the Positive Predictive Value (PPV) from a generic score vector.
    
    ‘min_separation’: minimum separation between the residues to be considered as a possible non-trivial contact. Default value is 6.
    ‘filestruct’: path to the file containing the residue pair distances.
    
The ‘filestruct‘ is a file containing a list of (i, j, d_ij) where d_ij is the distance in Angstrom between the residues i and j.

# Examples
```
julia> out = trainer("file.fasta",20,n_epochs=100);
julia> s = score(out.m..., min_separation = 6)
julia > ppv = compute_PPV(s, "filestruct.dat")
```
"""
function compute_PPV(score::Vector{Tuple{Int,Int,Float64}}, filestruct::String; min_separation::Int = 6)
    dist = compute_residue_pair_dist(filestruct)
    return map(x->x[4], compute_referencescore(score, dist, mindist = min_separation))
end

"""
    compute_PPV(arnet::ArDCA.ArNet, arvar::ArDCA.ArVar, seqid::Int64, filestruct::String; pc::Float64=0.1,min_separation::Int=6)
Function to compute the Positive Predictive Value (PPV) from the autoregressive model ArNet and ArVar.

    ‘min_separation’: minimum separation between the residues to be considered as a possible non-trivial contact. Default value is 6.
    ‘filestruct’: path to the file containing the residue pair distances.
    ‘seqid’: wild-type sequence identifier within the MSA from which the epistatic score is computed.

The ‘filestruct‘ is a file containing a list of (i, j, d_ij) where d_ij is the distance in Angstrom between the residues i and j. \n

# Examples
```
julia> out = artrainer("file.fasta",20,n_epochs=100);
julia> ppv = compute_PPV(out.ArNet, out.ArVar, 1, "filestruct.dat")
```
"""
function compute_PPV(arnet::ArDCA.ArNet, arvar::ArDCA.ArVar, seqid::Int64, filestruct::String; pc::Float64=0.1,min_separation::Int=6)
    score = ArDCA.epistatic_score(arnet, arvar, seqid, pc = pc, min_separation = min_separation)
    return compute_PPV(score, filestruct, min_separation = min_separation)
end


"""
    compute_PPV(out::AttOut, filestruct::String, kwds...)
Function to compute the Positive Predictive Value (PPV) from the attention model output structure for single families.\n 
The ‘filestruct‘ is a file containing a list of (i, j, d_ij) where d_ij is the distance in Angstrom between the residues i and j. \n
# Examples
```
julia> out = trainer("file.fasta",20,n_epochs=100);
julia> ppv = compute_PPV(out, "filestruct.dat")
```

"""
function compute_PPV(out::AttOut, filestruct::String; kwds...) 
    
    if out.m === nothing 
        return compute_PPV(out.score, filestruct; kwds...)
    else 
        return compute_PPV(score(out), filestruct; kwds...)
    end
end
"""
    compute_PPV(out::AttOut, files_struct::Vector{String}, kwds...)
Function to compute the Positive Predictive Value (PPV) from the attention model output structure for multiple families.\n
The `files_struct` is a list of paths to the files containing the structures of each family. The structure of a family is given as a list of (i, j, dij_) where d_ij is the distance in Angstrom between the residues i and j.
# Examples
```
julia> out = multi_trainer(["file1.fasta", "file2.fasta"],100,32,[23,23]);
julia> ppvs = compute_PPV(out, ["filestruct1.dat", "filestruct2.dat"])
```

"""
function compute_PPV(out::AttOut, files_struct::Vector{String}; kwds...) 
    if out.m === nothing 
        return [compute_PPV(score, files_struct[i]; kwds...) for i in 1:length(files_struct)]
    else 
        return [compute_PPV(score(out.m.Qs[i], out.m.Ks[i], out.m.V), files_struct[i]; kwds...) for i in 1:length(files_struct)]
    end
end

function compute_actualPPV(filestruct;cutoff=8.0,min_separation=6)
    distances=readdlm(filestruct)
    L,_ = size(distances)
    l = 0
    trivial_contacts = 0
    for i in 1:L
        if distances[i,end]<=cutoff #originally it was [i,4]
            if abs(distances[i,1]-distances[i,2]) > min_separation
                l += 1
            else 
                trivial_contacts += 1
            end
        end
    end
    #println("l = $l")
    #println("L = $L")
    #println("trivial contacts = $trivial_contacts")
    x = zeros(l)
    fill!(x,1.0)
    scra = map(x->l/x,[l+1:(L-trivial_contacts);])
    return vcat(x,scra) 
    
end

"""
    attention_heads(Q,K,V; sym =false)
Function to compute the attention heads of the (Q,K,V) model. \n
If ‘sym=true’ the matrices are symmetrised. \n
"""
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

attention_heads(out::AttOutStd, sym = false) = attention_heads(out.m.Q, out.m.K, out.m.V, sym = sym)

"""
    k_matrix(Q,K,V, k, version;...)
Given the model (Q,K,V) matrices, this function returns an LxL matrix with k highest elements for each head by either averaging (version = mean) them or by taking the maximimum (version = maximum) for each (i,j) couple.\n 
Keyword arguments:
    sym: symmetrize the attention heads, default true
    APC: apply the Average Product Correction, default false
    sqr: square the attention heads, default false
"""
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
    
    k_matrix(out::AttOutStd, k, version; kwds...) = k_matrix(out.m..., k, version; kwds...)
    
    k_matrix(Q,K,V, version; sym = true, APC = false, sqr = false) = k_matrix(Q,K,V, size(Q,3)^2,version; sym = sym, APC = APC, sqr = sqr)
    
    k_score(Q,K,V, k, version; sym = true, APC = false, sqr = false) = score_from_matrix(k_matrix(Q,K,V, k, version, sym = sym, APC = APC, sqr = sqr)[1])
    
    k_score(M) = score_from_matrix(M)
    
    attention_PPV(M, structure) = compute_PPV(k_score(M), structure)
    
    attention_PPV(Q,K,V, k, version, structure; sym = true, APC = false, sqr = false) = compute_PPV(k_score(Q,K,V, k, version, sym = sym, APC = APC, sqr = sqr), structure)
    
    attention_PPV(Q,K,V, version, structure; sym = true, APC = false, sqr = false) = compute_PPV(k_score(Q,K,V, size(Q,3)^2, version, sym = sym, APC = APC, sqr = sqr), structure)
    
    """
        score_from_matrix(A::AbstracMatrix)
    Given a LxL matrix, representing the positional information of the alignment, this function computes its coupling score as a vector of tuples (i,j,score(i,j)).
    """
    score_from_matrix(A::AbstractMatrix) = sort([(j,i,A[j,i]) for i in 2:size(A,2) for j in 1:i-1], by = x->x[3], rev = true)
    