function optimfunwrapper(g::Vector,x::Vector, var::AttPlmVar)
    g === nothing && (g = zeros(Float64, length(x)))
    return pl_and_grad!(g, x, var)
end

function counter_to_index(l::Int, N::Int, d:: Int, Q::Int, H::Int; verbose::Bool=false)
    h::Int = 0
    if l <= H*N*d
        i::Int = ceil(l/(d*H))
        l = l-(d*H)*(i-1)
        m::Int = ceil(l/H)
        h = l-H*(m-1)
        verbose && println("h = $h \nm = $m \ni = $i")
        return h,m,i
    elseif H*N*d < l <= 2*H*N*d 
        l-=d*N*H
        j::Int = ceil(l/(d*H))
        l-=(d*H)*(j-1)
        n::Int = ceil(l/H)
        h = l-H*(n-1)
        verbose && println("h = $h \nn = $n \nj = $j")
        return h, n, j
    else
        l-=2*N*d*H
        b::Int = ceil(l/(Q*H))
        l-=(Q*H)*(b-1)
        a::Int = ceil(l/H)
        h = l-H*(a-1)
        verbose && println("h = $h \na = $a \nb = $b \n")
        return h, a, b
    end
end

function logsumexp(a::AbstractArray{<:Real}; dims=1)
    m = maximum(a; dims=dims)
    return m + log.(sum(exp.(a .- m); dims=dims))
end

function sumexp(a::AbstractArray{<:Real};dims=1)
    m = maximum(a; dims=dims)
    return sum(exp.(a .- m ).*exp.(m); dims=dims)
end

function ReadFasta(filename::AbstractString,max_gap_fraction::Real, theta::Any, remove_dups::Bool)

    Z = read_fasta_alignment(filename, max_gap_fraction)
    if remove_dups
        Z, _ = remove_duplicate_sequences(Z)
    end

    N, M = size(Z)
    q = round(Int,maximum(Z))

    q > 32 && error("parameter q=$q is too big (max 31 is allowed)")
    W , Meff = compute_weights(Z,q,theta)
    rmul!(W, 1.0/Meff)
    Zint=round.(Int,Z)
    return W, Zint,N,M,q
end

function L2Tensor(matrix::Array{T,3}) where T <: Float64
    L2 = 0.0
    for x in matrix
        L2 += x*x 
    end
    return L2
end

function reshapetensor(J::Array{Float64, 4}, N::Int)
    newJ = Array{Float64, 3}[]
    for i in 1:N-1 
        push!(newJ,J[i+1,1:i,:,:])
    end
    return newJ
end

function computep0(var::AttPlmVar)
    W = var.W
    Z = var.Z 
    q = var.q 
    p0 = zeros(q)
    for i in 1:length(W)
        p0[Z[1, i]] += W[i]
    end
    p0
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

function sample(msamples::Int, J::Array{Array{Float64,3},1}, p0::Vector{Float64})
    q = length(p0)
    N = length(J) # here N is N-1 !!
    idxperm = 1:N
    backorder = sortperm(idxperm)
    res = Matrix{Int}(undef, N + 1, msamples)
    Threads.@threads for i in 1:msamples
        totH = Vector{Float64}(undef, q)
        sample_z = Vector{Int}(undef, N + 1)
        sample_z[1] = wsample(1:q, p0)
        for site in 1:N
            Js = J[site]
            @avx for i in 1:site
                for a in 1:q
                    totH[a] += Js[a, sample_z[i], i]
                end
            end
            p = softmax(totH)
            sample_z[site+1] = wsample(1:q, p)
        end
        res[:, i] .= sample_z
    end
    res
end
