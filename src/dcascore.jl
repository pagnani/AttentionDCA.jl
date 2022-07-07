function compute_PPV(filestruct::String, W,V; min_separation::Int=1, verbose=true, kwds...)
    H,N,N = size(W)
    score=compute_dcascore(W, V, min_separation=min_separation)
    dist = compute_residue_pair_dist(filestruct)
    roc = compute_referencescore(score, dist; kwds...)
    verbose && println("PPV(N) = ", roc[N][end], " PPV(N/5) = ", roc[div(N,5)][end])
    return roc
end

function compute_residue_pair_dist(filedist::String)
    d = readdlm(filedist)
    return Dict((round(Int,d[i,1]),round(Int,d[i,2])) => d[i,4] for i in 1:size(d,1) if d[i,4] != 0)
end

function compute_referencescore(score,dist::Dict; mindist::Int=6, cutoff::Number=8.0)
    nc2 = length(score)
    #nc2 == size(d,1) || throw(DimensionMismatch("incompatible length $nc2 $(size(d,1))"))
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
        if sitej - sitei > mindist 
            ctrtot += 1
            if dij < cutoff
                ctr += 1
            end
            push!(out,(sitei,sitej, plmscore, ctr/ctrtot))
        end
    end 
    out
end

function compute_dcascore(W, V; min_separation::Int=6)
    
    H,L,L = size(W)
    H,q,q = size(V)
    Wsf_site = softmax(W, dims=2)
    @tullio Jtens[a, b, r, i] := Wsf_site[h, i, r] * V[h, a, b] * (i != r)

    Jt = 0.5 * (Jtens + permutedims(Jtens,[2,1,4,3]))

    # Jmat = reshape(permutedims(Jtens,[1,4,2,3]),L*q,L*q)

    # Jmatsym = (Jmat .+ Jmat') ./ 2
    #Jt = permutedims(reshape(Jtenssymm, q, L, q, L), [1, 3, 4, 2]) 
    ht = zeros(eltype(Jt), q, L)
    Jzsg, _ = gauge(Jt, ht, ZeroSumGauge())
    FN = compute_fn(Jzsg)
    FNapc = correct_APC(FN)
    return compute_ranking(FNapc, min_separation)
end


function compute_dcascore_fa(Q, K, V; min_separation::Int=6)
    
    H,d,L = size(Q)
    H,d,L = size(K)
    H,q,q = size(V)
    @tullio W[h,i,j] := Q[h,d,i]*K[h,d,j] 
    W = softmax(W, dims=3)
    @tullio Jtens[a, b, r, i] := W[h, r, i] * V[h, a, b] * (i != r) (i in 1:L)

    Jt = 0.5 * (Jtens + permutedims(Jtens,[2,1,4,3]))

    # Jmat = reshape(permutedims(Jtens,[1,4,2,3]),L*q,L*q)

    # Jmatsym = (Jmat .+ Jmat') ./ 2
    #Jt = permutedims(reshape(Jtenssymm, q, L, q, L), [1, 3, 4, 2]) 
    ht = zeros(eltype(Jt), q, L)
    Jzsg, _ = gauge(Jt, ht, ZeroSumGauge())
    FN = compute_fn(Jzsg)
    FNapc = correct_APC(FN)
    return compute_ranking(FNapc, min_separation)
end


function parallel_compute_dcascore(W, V; min_separation::Int=6)
    
    H,L,L = size(W)
    H,q,q,N = size(V)
    Wsf_site = softmax(W, dims=2)
    @tullio Jtens[a, b, r, i] := Wsf_site[h, i, r] * V[h, a, b, r] * (i != r)

    Jt = 0.5 * (Jtens + permutedims(Jtens,[2,1,4,3]))

    # Jmat = reshape(permutedims(Jtens,[1,4,2,3]),L*q,L*q)

    # Jmatsym = (Jmat .+ Jmat') ./ 2
    #Jt = permutedims(reshape(Jtenssymm, q, L, q, L), [1, 3, 4, 2]) 
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


function compute_actualroc(filestruct;cutoff=8.0)
    distances=sort(readdlm(filestruct)[:,4])
    L = length(distances)
    l = 0
    for i in 1:L
        if distances[i]>cutoff
            l = i 
            break 
        end
    end
    L -= l
    x = zeros(L)
    fill!(x,1.0)
    scra = map(x->L/x,[L+1:L+l;])
    return vcat(x,scra) 
    
end

