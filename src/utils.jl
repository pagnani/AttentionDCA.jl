function softmax_notinplace(x::AbstractArray; dims = 1)
    dims > length(size(x)) && error("dims is too big, it can range from 1 to $(length(size(x)))")
    T = eltype(x)
    max_ = maximum(x; dims)
    out = similar(x)
    
    if all(isfinite, max_)
        @fastmath out = exp.(x .- max_)
    else
        @fastmath @. out = ifelse(isequal(max_,Inf), ifelse(isequal(x,Inf), 1, 0), exp(x - max_)) 
        #@fastmath @. out = ifelse(isequal(max_,-Inf), ifelse(isequal(x,-Inf),0,0), ifelse(isequal(max_,Inf), ifelse(isequal(x,Inf), 1, 0), exp(x - max_))) 
    end

    sums_ = sum(out; dims = dims)
    index = sums_ .== 0

    sums = sums_ + index
    
    return  out./sums
    # return out
end

function hardmax(A::AbstractArray; dims::Int=1)
    max_index = argmax(A, dims=dims)
    out = zeros(Int, size(A))
    for i in max_index
        out[i] = 1
    end
    return out
end


function logsumexp(a::AbstractArray{<:Real}; dims=1)
    m = maximum(a; dims=dims)
    return m + log.(sum(exp.(a .- m); dims=dims))
end

function sumexp(a::AbstractArray{<:Real};dims=1)
    m = maximum(a; dims=dims)
    return sum(exp.(a .- m ).*exp.(m); dims=dims)
end

function ReadFasta(filename::AbstractString,max_gap_fraction::Real, theta::Any, remove_dups::Bool;verbose=true)

    Z = read_fasta_alignment(filename, max_gap_fraction)
    if remove_dups
        Z, _ = remove_duplicate_sequences(Z,verbose=verbose)
    end

    N, M = size(Z)
    q = round(Int,maximum(Z))

    q > 32 && error("parameter q=$q is too big (max 31 is allowed)")
    W , Meff = compute_weights(Z,q,theta,verbose=verbose)
    rmul!(W, 1.0/Meff)
    Zint=round.(Int,Z)
    return W,Zint,N,M,q
end

function quickread(fastafile; moreinfo=false)  
    Weights, Z, N, M, _ = ReadFasta(fastafile, 0.9, :auto, true, verbose = false);
    moreinfo && return Weights, Z, N, M
    return Z, Weights
end

function L2Tensor(matrix::AbstractArray{T}) where T <: Real
    L2 = 0.0
    for x in matrix
        L2 += x*x 
    end
    return L2
end

function reshapetensor(J::Array{Float64, 4}, N::Int, q::Int)
    newJ = Array{Float64, 3}[]
    _J = zeros(Float64, N, q, q)
    for i in 1:N-1 
        _J = J[i+1,1:i,:,:]
        @tullio Jscra[a,b,j] := _J[j,a,b]
        push!(newJ,Jscra)
    end
    return newJ
end

function computep0(D; q::Int = 21)
    p0 = zeros(q)
    for i in 1:length(D[2])
        p0[D[1][1, i]] += D[2][i]
    end
    return p0
end

function compute_empirical_freqs(Z::AbstractArray{Ti,2}, W::AbstractVector{Float64}, q::Ti) where {Ti<:Integer}
    N, M = size(Z)
    f = zeros(q, N)
    @inbounds for i in 1:N
        for s in 1:M
            f[Z[i, s], i] += W[s]
        end
    end
    f
end

"""
    compute_freq(Z::Matrix,W::Vector{Float64})
Function that returns a tuple with the single and pairwise weighted frequencies of the alignment Z and weigth vector W.

# Examples
```
julia> f1, f2 = compute_freq(Z,W)
```
"""
compute_freq(Z::Matrix,W::Vector{Float64}) = compute_weighted_frequencies(Matrix{Int8}(Z), W)


"""
    compute_freq(Z::Matrix)
Function that returns a tuple with the single and pairwise frequencies of the alignment Z. The weight vector is set to 1/M, where M is the number of sequences in the alignment.
"""
compute_freq(Z::Matrix) = compute_weighted_frequencies(Matrix{Int8}(Z), fill(1/size(Z,2), size(Z,2)))


function entropy(Z::AbstractArray{Ti,2}, W::AbstractVector{Float64}) where {Ti<:Integer}
    N,_ = size(Z)
    q = maximum(Z)
    f = compute_empirical_freqs(Z, W, q)
    S = zeros(N)
    @inbounds for i in 1:N
        _s = 0.0
        for a in 1:q
            _s -= f[a, i] > 0 ? f[a, i] * log(f[a, i]) : 0.0
        end
        S[i] = _s
    end
    S
end

# softmax(x::AbstractArray{T}; dims = 1) where {T} = softmax!(similar(x, float(T)), x; dims)

# softmax!(x::AbstractArray; dims = 1) = softmax!(x, x; dims)

# function softmax!(out::AbstractArray{T}, x::AbstractArray; dims = 1) where {T}
#     max_ = maximum(x; dims)
#     if all(isfinite, max_)
#         @fastmath out .= exp.(x .- max_)
#     else
#         @fastmath @. out = ifelse(isequal(max_,Inf), ifelse(isequal(x,Inf), 1, 0), exp(x - max_))
#     end
#     out ./= sum(out; dims)
# end

function L2reg(Q::AbstractArray{Float64,3},K::AbstractArray{Float64,3},V::AbstractArray{Float64,3},lambda)
    _,d,N = size(Q)
    _,q,q = size(V)

    numpar = N*(N-1)*q*q
    @tullio sf[i, j, h] := Q[h,d,i]*K[h,d,j]
    sf = softmax_notinplace(sf./sqrt(d),dims=2) 

    @tullio J[i,j,a,b] := sf[i,j,h]*V[h,a,b]*(j!=i)
    
    l2 = L2Tensor(J)

    λ = lambda / numpar
    return λ*l2
end

function L2reg(J::AbstractArray{Float64,4}, lambda)
    N,N,q,q = size(J)
    numpar = N*(N-1)*q*q
    l2 = L2Tensor(J)

    λ = lambda / numpar
    return λ*l2
end

