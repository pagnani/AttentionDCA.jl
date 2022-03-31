function optimfunwrapper(x::Vector, g::Vector, var)
    g === nothing && (g = zeros(Float64, length(x)))
    return pl_and_grad!(g, x,  var)
end

function optimfunwrapperJreg(x::Vector, g::Vector, var)
    g === nothing && (g = zeros(Float64, length(x)))
    return pl_and_grad_jreg!(g, x, var)
end

function counter_to_index(l::Int, N::Int, Q::Int, H::Int; verbose::Bool=false)
    h::Int = 0
    if l <= H*N*N
            j::Int = ceil(l/(N*H))
            l = l-(N*H)*(j-1)
            i::Int = ceil(l/H)
            h = l-H*(i-1)
            verbose && println("h = $h \ni = $i \nj = $j\n")
            return h,i,j
    else
            l-=N*N*H
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

function L2Tensor(matrix)
    L2 = 0.0
    for x in matrix
        L2 += x*x 
    end
    return L2
end