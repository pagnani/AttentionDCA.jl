import Flux:@functor,batched_adjoint,batched_mul,Dense
struct SelfAttention{T3}
    d1::Int
    d2::Int
    H::Int
    Q::T3
    K::T3
    V::T3
end

function SelfAttention(d1::Int, d2::Int, H::Int; rt::Type{T}=Float32) where {T<:AbstractFloat}
    Q, K, V = randn(rt, d1, d2, H) * T(1e-3), randn(rt, d1, d2, H) * T(1e-3), randn(rt, d1, d1, H) * T(1e-3)
    SelfAttention(d1, d2, H, Q, K, V)
end

Flux.trainable(sa::SelfAttention) = (sa.Q,sa.K,sa.V)
Base.show(io::IO, sa::SelfAttention) = print(io, "SelfAttention{$(eltype(sa.Q))}[din = $(sa.d1), dou = $(sa.d2), H = $(sa.H) numpara = $(sum(length.(Flux.params(sa))))]")
@functor SelfAttention

function prepare_network(filename::String; 
    max_gap_fraction::Float64=1.0, 
    theta::Union{Symbol,Float64}=:auto, 
    remove_dups::Bool=true,
    rt::Type{T}=Float64,
    dou::Int=21,
    din::Int=21,
    H::Int= 32
) where {T<:AbstractFloat}
    W, Z, _, _, _ = ArDCA.read_fasta(filename, max_gap_fraction, theta, remove_dups)
    Z = rt.(one_hot(Int.(Z),q=din))
    W = rt.(W)
    return Z,W,SelfAttention(din, dou, H)
end

function cross_attention(sa::SelfAttention,Z)
    WQ, WK, WV = sa.Q, sa.K, sa.V
    @tullio Q[i, d2, m, h] := Z[d1, i, m] * WQ[d1, d2, h]
    @tullio K[i, d2, m, h] := Z[d1, i, m] * WK[d1, d2, h]
    @tullio V[i, d2, m, h] := Z[d1, i, m] * WV[d1, d2, h]
    @tullio A[i, j, h] := Q[i, d, m, h] * K[j, d, m, h] * (i!=j)
    A = Flux.softmax(A / size(Z, 3), dims=2)
end

function (sa::SelfAttention)(Z::AbstractArray)
    WQ, WK, WV, H = sa.Q, sa.K, sa.V, sa.H
    @tullio Q[i, d2, m, h] := Z[d1, i, m] * WQ[d1, d2, h]
    @tullio K[i, d2, m, h] := Z[d1, i, m] * WK[d1, d2, h]
    @tullio V[i, d2, m, h] := Z[d1, i, m] * WV[d1, d2, h]
    @tullio A[i, j, h] := Q[i, d, m, h] * K[j, d, m, h]
    A = Flux.softmax(A / size(Z,3), dims=2)
    @tullio Y[d, i, m] := A[i, j, h] * V[j, d, m, h] * (i != j)
    return Flux.softmax(Y / H, dims=1)
end

function myloss_andrea(sa,Z,W)
    sw = sum(W)
    Y = sa(Z)
    @tullio loss := Z[d,i,m] * log(Y[d, i, m]) * W[m]
    return -loss/sw
end

function myl2loss(sa::SelfAttention)
    return sum(abs2,sa.Q) + sum(abs2,sa.K) + sum(abs2,sa.V)
end

function trainnet!(sa,Z, W::Vector{T};
    niter::Int=100,
    λ::Real=0.01,
    Δt::Int=10,
    batchsize::Int=1000,
    η::Real=0.001,
    timeout::Real=10
) where T<:AbstractFloat
    local_loss(Z, W) = myloss_andrea(sa, Z, W) + λ * myl2loss(sa)
    λ,η = T(λ), T(η)
    evalcb = (it) -> let        
        e1 = myloss_andrea(sa,Z,W)
        e2 = λ * myl2loss(sa)
        @info  "Epoch $it total-loss $(e1+e2) kld $e1 l2loss $e2"
    end 
    opt = Flux.Optimise.ADAM(η) 
    data = Flux.DataLoader((Z, W), batchsize=batchsize)
    for it in 1:niter
        Flux.train!(local_loss, Flux.params(sa), data, opt)
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

function compute_residue_pair_dist(filedist::String; threshold::Real=7.0)
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


function first_k_pairs(sa::SelfAttention,Z,k; apc::Bool=true)
    fun = apc ? ArDCA.correct_APC : x->x
    A=cross_attention(sa::SelfAttention,Z)
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