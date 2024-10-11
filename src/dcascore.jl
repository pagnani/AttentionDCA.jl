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

"""
    score(Jtens; min_separation::Int=6)
Function to compute the Frobenious contact score directly from the trained model m = (Q,K,V)
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
Function to compute the Positive Predictive Value (PPV) from a generic score vector. \n
Argument ‘min_separation’ is the minimum separation between the residues to be considered as a possible non-trivial contact. Default value is 6. \n
Argument ‘filestruct’ is the path to the file containing the residue pair distances. \n
"""
function compute_PPV(score::Vector{Tuple{Int,Int,Float64}}, filestruct::String; min_separation::Int = 6)
    dist = compute_residue_pair_dist(filestruct)
    return map(x->x[4], compute_referencescore(score, dist, mindist = min_separation))
end

"""
    compute_PPV(arnet::ArDCA.ArNet, arvar::ArDCA.ArVar, seqid::Int64, filestruct::String; pc::Float64=0.1,min_separation::Int=6)
Function to compute the Positive Predictive Value (PPV) from the autoregressive model ArNet and ArVar. \n
Argument ‘min_separation’ is the minimum separation between the residues to be considered as a possible non-trivial contact. Default value is 6. \n
Argument ‘filestruct’ is the path to the file containing the residue pair distances. \n
Argument ‘seqid’ is the wild-type sequence identifier within the MSA from which the epistatic score is computed. \n
"""
function compute_PPV(arnet::ArDCA.ArNet, arvar::ArDCA.ArVar, seqid::Int64, filestruct::String; pc::Float64=0.1,min_separation::Int=6)
    score = ArDCA.epistatic_score(arnet, arvar, seqid, pc = pc, min_separation = min_separation)
    return compute_PPV(score, filestruct, min_separation = min_separation)
end

function compute_actualPPV(filestruct;cutoff=8.0,min_separation=6)
    distances=readdlm(filestruct)
    L,_ = size(distances)
    l = 0
    trivial_contacts = 0
    for i in 1:L
        if distances[i,end]<cutoff #originally it was [i,4]
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
