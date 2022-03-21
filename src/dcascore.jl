# function computescore(score,filedist::String; mindist::Int=4, cutoff::Float64=7.0)
#     d = readdlm(filedist)
#     dist = Dict((round(Int,d[i,1]),round(Int,d[i,2])) => d[i,4] for i in 1:size(d,1))
#     nc2 = length(score)
#     #nc2 == size(d,1) || throw(DimensionMismatch("incompatible length $nc2 $(size(d,1))"))
#     out = Tuple{Int,Int,Float64,Float64}[]
#     ctrtot = 0
#     ctr = 0
#     for i in 1:nc2
#         sitei,sitej,plmscore = score[i][1],score[i][2], score[i][3]
#         dij = if haskey(dist,(sitei,sitej)) 
#             dist[(sitei,sitej)]
#         else
#            continue
#         end
#         if sitej - sitei > mindist
#             ctrtot += 1
#             if dij < cutoff
#                 ctr += 1
#             end
#             push!(out,(sitei,sitej, plmscore, ctr/ctrtot))
#         end
#     end 
#     out
# end

using DelimitedFiles
function compute_PPV(filestruct::String, score; verbose=true, kwds...)
    L= maximum(map(x->x[2],score))
    dist = compute_residue_pair_dist(filestruct)
    roc = compute_referencescore(score, dist; kwds...)
    verbose && println("PPV(L) = ", roc[L][end], " PPV(L/5) = ", roc[div(L,5)][end])
    return roc
end


function compute_residue_pair_dist(filedist::String)
    d = readdlm(filedist)
    return Dict((round(Int,d[i,1]),round(Int,d[i,2])) => d[i,4] for i in 1:size(d,1))
end

function compute_referencescore(score,dist::Dict; mindist::Int=4, cutoff::Number=7.0)
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
