import Flux:@functor,batched_adjoint,batched_mul,Dense
import Distributions: wsample 

function onehot2cat(Z)
    d,L,M = size(Z)
    Zcat = zeros(Int,L,M)
    for m in 1:M
        for i in 1:L
            Zcat[i,m] = argmax(view(Z,:,i,m))
        end
    end
    return Matrix(Zcat)
end

struct SelfAttention{T3}
    demb::Int # input dimension
    datt::Int # attention internal dimension
    dout::Int # output dimension
    H::Int
    Q::T3
    K::T3
    V::T3
    O::T3
    mask::Union{Symbol, Nothing}
end

struct SelfAttention_Tied{T3}
    demb::Int # input dimension
    datt::Int # attention internal dimension
    dout::Int # output dimension
    H::Int
    Q::T3
    K::T3
    V::T3
    O::T3
    mask::Union{Symbol, Nothing}
end

function SelfAttention(demb::Int, datt::Int, dout::Int, H::Int; mask::Union{Symbol, Nothing} = nothing, rt::Type{T}=Float32,ongpu::Bool=false, init_fun = randn, init_scale = 1.0e-3) where {T<:AbstractFloat}
    fun = ongpu ? gpu : x->x

    mask ∈ [nothing, :causal, :diagonal] || error("The mask method is not valid, only :causal and :diagonal are allowed, or nothing.")

    Q, K, V, O = init_fun(rt, demb, datt, H) * T(init_scale), init_fun(rt, demb, datt, H) * T(init_scale), init_fun(rt, demb, demb, H) * T(init_scale), init_fun(rt, demb, dout, H) * T(init_scale)
    SelfAttention(demb, datt, dout, H, fun(Q), fun(K), fun(V), fun(O), mask)
end

Flux.trainable(sa::SelfAttention) = (sa.Q,sa.K,sa.V,sa.O)

function Base.show(io::IO, sa::SelfAttention)
    ongpu = typeof(sa.V) <: CuArray
    print(io, "SelfAttention{$(eltype(sa.Q))}[demb=$(sa.demb), datt=$(sa.datt), dout=$(sa.dout), H=$(sa.H), numpara=$(sum(length.(Flux.params(sa)))), onpgu=$ongpu, mask=$(sa.mask)]")
end

@functor SelfAttention

function SelfAttention_Tied(demb::Int, datt::Int, dout::Int, H::Int;mask::Union{Symbol, Nothing}, rt::Type{T}=Float32,ongpu::Bool=false, init_fun = randn, init_scale = 1.0e-3) where {T<:AbstractFloat}
    fun = ongpu ? gpu : x->x

    mask ∈ [nothing, :causal, :diagonal] || error("The mask method is not valid, only :causal and :diagonal are allowed, or nothing.")

    Q, K, V, O = init_fun(rt, demb, datt, H) * T(init_scale), init_fun(rt, demb, datt, H) * T(init_scale), init_fun(rt, demb, demb, H) * T(init_scale), init_fun(rt, demb, dout, H) * T(init_scale)
    SelfAttention_Tied(demb, datt, dout, H, fun(Q), fun(K), fun(V), fun(O), mask)
end

Flux.trainable(sa::SelfAttention_Tied) = (sa.Q,sa.K,sa.V,sa.O)

function Base.show(io::IO, sa::SelfAttention_Tied)
    ongpu = typeof(sa.V) <: CuArray
    print(io, "SelfAttention_Tied{$(eltype(sa.Q))}[demb=$(sa.demb), datt=$(sa.datt), dout=$(sa.dout), H=$(sa.H), numpara=$(sum(length.(Flux.params(sa)))), onpgu=$ongpu, mask=$(sa.mask)]")
end

@functor SelfAttention_Tied

function prepare_network(filename::String; 
    max_gap_fraction::Float64=1.0, 
    theta::Union{Symbol,Float64}=:auto, 
    remove_dups::Bool=true,
    rt::Type{T}=Float64,
    datt::Int=21,
    demb::Int=21,
    dout::Int=demb,
    H::Int= 32,
    tied::Bool=false,
    mask::Union{Symbol, Nothing} = nothing,
    ongpu::Bool=false) where {T<:AbstractFloat}


    fun = ongpu ? gpu : x->x
    W, Z, _, _, _ = ArDCA.read_fasta(filename, max_gap_fraction, theta, remove_dups)
    Z = rt.(one_hot(Int.(Z),q=demb))
    W = rt.(W)
    if tied
        return fun(Z),fun(W),SelfAttention_Tied(demb, datt, dout, H, ongpu=ongpu, mask=mask)
    else
        return fun(Z),fun(W),SelfAttention(demb, datt, dout, H, ongpu=ongpu, mask=mask)
    end
end


function cross_attention(sa::SelfAttention_Tied,Z)
    WQ, WK, WV, mask = sa.Q, sa.K, sa.V, sa.mask
    msk = build_mask(mask, size(Z,2))
    @tullio Q[i, d2, m, h] := Z[d1, i, m] * WQ[d1, d2, h]
    @tullio K[i, d2, m, h] := Z[d1, i, m] * WK[d1, d2, h]
    @tullio V[i, d2, m, h] := Z[d1, i, m] * WV[d1, d2, h]
    @tullio A[i, j, h] := Q[i, d, m, h] * K[j, d, m, h]*(i!=j)
    A = apply_mask(A, msk)
    return softmax_notinplace(A / size(Z, 3), dims=2)
end

function cross_attention(sa::SelfAttention,Z)
    WQ, WK, WV, mask = sa.Q, sa.K, sa.V, sa.mask
    msk = build_mask(mask, size(Z,2))
    @tullio Q[i, d2, m, h] := Z[d1, i, m] * WQ[d1, d2, h]
    @tullio K[i, d2, m, h] := Z[d1, i, m] * WK[d1, d2, h]
    @tullio V[i, d2, m, h] := Z[d1, i, m] * WV[d1, d2, h]
    @tullio A[i, j, m, h] := Q[i, d, m, h] * K[j, d, m, h]*(i!=j)
    A = apply_mask(A, msk)
    return softmax_notinplace(A, dims=2)
end

function (sa::SelfAttention_Tied)(Z::AbstractArray, W::AbstractVector)
    WQ, WK, WV, WO, H, mask = sa.Q, sa.K, sa.V, sa.O, sa.H, sa.mask
    sw = inv(sum(W))
    msk = build_mask(mask, size(Z, 2))
    @tullio Q[i, d2, m, h] := Z[d1, i, m] * WQ[d1, d2, h] * W[m] * $sw
    @tullio K[i, d2, m, h] := Z[d1, i, m] * WK[d1, d2, h] * W[m] * $sw
    @tullio V[i, d2, m, h] := Z[d1, i, m] * WV[d1, d2, h] * W[m] * $sw
    @tullio A[i, j, h] := Q[i, d, m, h] * K[j, d, m, h]
    A = apply_mask(A, msk)
    A = softmax_notinplace(A / size(Z, 3), dims=2)
    @tullio _Y[d, i, m, h] := A[i, j, h] * V[j, d, m, h]*(i != j)
    @tullio Y[dout,i,m] := _Y[demb,i,m,h] * WO[demb,dout,h]
    return Flux.softmax(Y / H, dims=1)
end

function (sa::SelfAttention)(Z::AbstractArray, W::AbstractVector)
    WQ, WK, WV, WO, H, mask = sa.Q, sa.K, sa.V, sa.O, sa.H, sa.mask
    sw = inv(sum(W))
    msk = build_mask(mask, size(Z, 2))
    @tullio Q[i, d2, m, h] := Z[d1, i, m] * WQ[d1, d2, h] * W[m] * $sw
    @tullio K[i, d2, m, h] := Z[d1, i, m] * WK[d1, d2, h] * W[m] * $sw
    @tullio V[i, d2, m, h] := Z[d1, i, m] * WV[d1, d2, h] * W[m] * $sw
    @tullio A[i, j, m, h] := Q[i, d, m, h] * K[j, d, m, h]
    A = apply_mask(A, msk)
    A = softmax_notinplace(A, dims=2)
    @tullio _Y[d, i, m, h] := A[i, j, m, h] * V[j, d, m, h]*(i != j)
    @tullio Y[dout, i, m] := _Y[demb,i,m,h] * WO[demb,dout,h]
    return Flux.softmax(Y / H, dims=1)
end


function (sa::SelfAttention_Tied)(Z::AbstractArray)
    WQ, WK, WV, WO, H, mask = sa.Q, sa.K, sa.V, sa.O, sa.H, sa.mask
    #sw = inv(sum(W))
    msk = build_mask(mask, size(Z, 2))

    @tullio Q[i, d2, m, h] := Z[d1, i, m] * WQ[d1, d2, h]
    @tullio K[i, d2, m, h] := Z[d1, i, m] * WK[d1, d2, h]
    @tullio V[i, d2, m, h] := Z[d1, i, m] * WV[d1, d2, h]
    @tullio A[i, j, h] := Q[i, d, m, h] * K[j, d, m, h]
    A = apply_mask(A, msk)
    A = softmax_notinplace(A / size(Z, 3), dims=2)
    @tullio _Y[d, i, m, h] := A[i, j, h] * V[j, d, m, h]*(i!=j)
    @tullio Y[dout,i,m] := _Y[demb,i,m,h] * WO[demb,dout,h]
    return Flux.softmax(Y / H, dims=1)
end

function (sa::SelfAttention)(Z::AbstractArray)    WQ, WK, WV, WO, H, mask = sa.Q, sa.K, sa.V, sa.O, sa.H, sa.mask
    #sw = inv(sum(W))
    msk = build_mask(mask, size(Z, 2))

    @tullio Q[i, d2, m, h] := Z[d1, i, m] * WQ[d1, d2, h]
    @tullio K[i, d2, m, h] := Z[d1, i, m] * WK[d1, d2, h]
    @tullio V[i, d2, m, h] := Z[d1, i, m] * WV[d1, d2, h]
    @tullio A[i, j, m, h] := Q[i, d, m, h] * K[j, d, m, h]
    A = apply_mask(A, msk)
    A = softmax_notinplace(A, dims=2)
    @tullio _Y[d, i, m, h] := A[i, j, m, h] * V[j, d, m, h]*(i!=j)
    @tullio Y[dout, i, m] := _Y[demb,i,m,h] * WO[demb,dout,h]
    return Flux.softmax(Y / H, dims=1)
end

function build_mask(masktype::Union{Nothing, Symbol}, L::Int)
    if masktype == :causal
        mask = tril(ones(Bool, L,L))
    elseif masktype == :diagonal
        mask = Bool.(1 .- I(L))
    else
        mask = nothing
    end
   return mask
end

apply_mask(x, mask::Nothing) = x

function apply_mask(x, mask)
    neginf = typemin(eltype(x))
    ifelse.(mask, x, neginf)
end


# Tullio does not reduce to scalar on gpu. for CuArray we do the one below
# function myloss_andrea(sa,Z::AbstractArray{T,3},W::AbstractVector{T}) where {T<:AbstractFloat}
#     sw = sum(W)
#     Y = sa(Z)
#     @tullio loss[d] := Z[d,i,m] * log(Y[d, i, m]) * W[m]
#     return -sum(loss)/sw
# end

# ... scalar reduction in Tullio is ok for Array.
function KL_loss(sa::Union{SelfAttention, SelfAttention_Tied}, Z::Array{T,3}, W::Vector{T}) where {T<:AbstractFloat}
    sw = sum(W)
    Y = sa(Z)
    @tullio loss := Z[d, i, m] * log(Y[d, i, m]) * W[m]
    return -loss / sw
end

function L2_loss(sa::Union{SelfAttention, SelfAttention_Tied})
    return sum(abs2,sa.Q) + sum(abs2,sa.K) + sum(abs2,sa.V)
end

function L2_attention_loss(sa::Union{SelfAttention, SelfAttention_Tied},Z)
    A = cross_attention(sa,Z)
    return sum(abs2,A)
end

function trainnet!(sa::Union{SelfAttention, SelfAttention_Tied},Z, W::AbstractVector{T};
    niter::Int=100,
    λ::Real=0.01,
    reg_fun::Function = L2_loss,
    Δt::Int=10,
    batchsize::Int=1000,
    η::Real=0.001,
    timeout::Real=10
) where T<:AbstractFloat
    local_loss(sa, Z, W) = KL_loss(sa, Z, W) + λ * reg_fun(sa)
    λ,η = T(λ), T(η)
    evalcb = (it) -> let        
        e1 = KL_loss(sa,Z,W)
        e2 = λ * reg_fun(sa)
        @info  "Epoch $it total-loss $(e1+e2) kld $e1 l2loss $e2"
    end 
    opt = Flux.Optimise.ADAM(η) 
    #data = Flux.DataLoader((Z, W), batchsize=batchsize)
    for it in 1:niter
        loader = Flux.DataLoader((Z, W), batchsize=batchsize, shuffle=true)
        for (z,w) in loader
            grads = gradient(()->local_loss(sa,z,w), Flux.params(sa))
            Flux.Optimise.update!(opt, Flux.params(sa), grads)
        end
        if it % Δt == 0
            evalcb(it)
        end
    end
    sa
end

function klargest_indexes(m, k)
    ci = CartesianIndices(size(m))
    p = partialsortperm(vec(m), 1:k; rev=true)
    ci[p]
end

function residue_pair_dist(filedist::String; threshold::Real=7.0)
    d = readdlm(filedist)
    dist = Dict((round(Int, d[i, 1]), round(Int, d[i, 2])) => d[i, 4] for i in axes(d,1))
    N = maximum(x->x[2], keys(dist))
    distmat = zeros(N,N)
    bbone = filter(x->x[2]<threshold,dist)
    for (i,j) in keys(bbone)
        distmat[i,j] = 1.0
        distmat[j,i] = 1.0
    end
    distmat
end


function first_k_pairs(sa::SelfAttention_Tied,Z,k; apc::Bool=true)
    fun = apc ? ArDCA.correct_APC : x->x
    A = cross_attention(sa::SelfAttention_Tied,Z) |> cpu
    res = zeros(size(A,1),size(A,2))
    for h in axes(A,3)
        vAh =A[:, :, h]
        cis = klargest_indexes(fun(vAh), k)
        vm = vAh[cis]
        ctr = 0
        for ci in cis
            ctr += 1 
            res[ci] += vm[ctr]
        end
    end 
    res
end

function first_k_pairs(sa::SelfAttention,Z,k; apc::Bool=true)
    fun = apc ? ArDCA.correct_APC : x->x
    A = cross_attention(sa::SelfAttention,Z) |> cpu
    res = zeros(size(A,1),size(A,2))
    for h in axes(A,4)
        for m in axes(A,3)
            vAh = A[:, :, m, h]
            cis = klargest_indexes(fun(vAh), k)
            vm = vAh[cis]
            ctr = 0
            for ci in cis
                ctr += 1 
                res[ci] += vm[ctr]
            end
        end
    end 
    res/size(A,4)
end

function loss_grad(sa,Z,W)
    gradient(()->myloss_andrea(sa,Z,W), Flux.params(sa))
end

function first_k_pairs(A::Array{T,3},k; apc::Bool = true) where {T<:AbstractFloat}
    fun = apc ? ArDCA.correct_APC : x->x
    res = zeros(size(A,1),size(A,2))
    for h in axes(A,3)
        vAh = A[:, :, h]
        cis = klargest_indexes(fun(vAh), k)
        vm = vAh[cis]
        ctr = 0
        for ci in cis
            ctr += 1 
            res[ci] += vm[ctr]
        end
    end 
    res
end

function first_k_pairs(A::Array{T,4},k; apc::Bool = true) where {T<:AbstractFloat}
    fun = apc ? ArDCA.correct_APC : x->x
    res = zeros(size(A,1),size(A,2))
    for h in axes(A,4)
        for m in axes(A,3)
            vAh = A[:, :, m, h]
            cis = klargest_indexes(fun(vAh), k)
            vm = vAh[cis]
            ctr = 0
            for ci in cis
                ctr += 1 
                res[ci] += vm[ctr]
            end
        end
    end 
    res/size(A,4)
end

function initial_freq(Z,W)
    q = maximum(Z)
    p0 = zeros(q)
    for i in eachindex(W)
        p0[Z[1, i]] += W[i]
    end
    return p0/sum(W)
end

function freq(Z,W,s)
    q = maximum(Z)
    p0 = zeros(q)
    for i in eachindex(W)
        p0[Z[s, i]] += W[i]
    end
    return p0/sum(W)
end

function sampler(sa::Union{SelfAttention,SelfAttention_Tied}, msamples::Int, Z,W)
    f0 = initial_freq(Z,W)
    L,M = size(Z)
    q = length(f0)

    res = Array{Int}(undef, q, L, msamples)
    Threads.@threads for i in 1:msamples
        sample_z = zeros(Int, q, L, 1)
        sample_z[wsample(1:q, f0), 1, 1] = 1
        for site in 2:L
            y = sa(sample_z)
            sample_z[wsample(1:q, y[:,site,1]), site, 1] = 1
        end
        res[:, :, i] .= sample_z[:,:,1]
    end
    return res
end