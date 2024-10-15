abstract type AttOut end 


struct AttOutStd <: AttOut
    m::Union{Nothing, NamedTuple}
    score::Union{Nothing, Vector{Tuple{Int64, Int64, Float64}}}
end

function Base.show(io::IO, attoutstd::AttOutStd)
    @extract attoutstd : m score
    if m !== nothing 
        H,d,L = size(m.Q)
        _,q,_ = size(m.V)
    end
    if score === nothing && m!==nothing
        print(io,"Standard single-shot version, AttOutStd m = (Q,K,V), [L=$L H=$H d=$d, q=$q]")
    elseif score !== nothing && m === nothing
        print(io,"Standard statistical version, Score length = $(length(score))")
    end
end

struct AttOutAr <: AttOut
    m::Union{Nothing, NamedTuple}
    ArNet::Union{Nothing, ArDCA.ArNet} 
    ArVar::Union{Nothing, ArDCA.ArVar}
    score::Union{Nothing, Vector{Tuple{Int64, Int64, Float64}}}
end

function Base.show(io::IO, attoutar::AttOutAr)
    @extract attoutar : m ArNet ArVar score
    if m !== nothing 
        H,d,L = size(m.Q)
        _,q,_ = size(m.V)
    end
    if score === nothing && m!==nothing
        print(io,"Autoregressive single-shot version, AttOutAr.m = (Q,K,V), [L=$L H=$H d=$d, q=$q]")
    elseif score !== nothing && m === nothing
        print(io,"Autoregressive statistical version, AttOutAr.score length = $(length(score))")
    end
end

struct AttOutMulti <: AttOut
    m::Union{Nothing, NamedTuple}
    score::Union{Nothing, Vector{Vector{Tuple{Int64, Int64, Float64}}}}
end

function Base.show(io::IO, attoutmulti::AttOutMulti)
    @extract attoutmulti : m score
    if score === nothing && m!==nothing
        F = length(m.Qs)
        Ls = [size(m.Qs[i],3) for i in 1:F]
        print(io,"Multi-family single-shot version, AttOutMulti.m = (Qs,Ks,V), $F families of lengths $(Ls)")
    elseif score !== nothing && m === nothing
        print(io,"Multi-family statistical version, AttOutMulti.score lengths = $([length(score[i]) for i in axes(score)])")
    end
end