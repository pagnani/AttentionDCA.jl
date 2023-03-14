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

function score(Q, K, V; min_separation::Int=6)
    
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


function compute_actualPPV(filestruct;cutoff=8,min_separation=6)
    distances=readdlm(filestruct)
    L,_ = size(distances)
    l = 0
    trivial_contacts = 0
    for i in 1:L
        if distances[i,4]<cutoff
            if abs(distances[i,1]-distances[i,2]) > min_separation
                l += 1
            else 
                trivial_contacts += 1
            end
        end
    end
    println("l = $l")
    println("L = $L")
    println("trivial contacts = $trivial_contacts")
    x = zeros(l)
    fill!(x,1.0)
    scra = map(x->l/x,[l+1:(L-trivial_contacts);])
    return vcat(x,scra) 
    
end

function compute_PPV(Out::AttPlmOut,filestruct)
    @extract Out: Q K V
    
    dist = compute_residue_pair_dist(filestruct)
    _score = score(Q, K, V) 
    return map(x->x[4], compute_referencescore(_score, dist))

end 
function compute_PPV(Out::FieldAttPlmOut,filestruct)
    @extract Out: Q K V
    
    dist = compute_residue_pair_dist(filestruct)
    _score = score(Q, K, V) 
    return map(x->x[4], compute_referencescore(_score, dist))

end 

function compute_PPV(score::Vector{Tuple{Int,Int,Float64}}, filestruct::String)
    dist = compute_residue_pair_dist(filestruct)
    return map(x->x[4], compute_referencescore(score, dist))
end

