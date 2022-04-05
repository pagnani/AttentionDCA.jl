struct PlmAlg
    method::Symbol
    verbose::Bool
    epsconv::Float64
    maxit::Int
end

struct PlmOut
    pslike::Union{Vector{Float64},Float64}
    Wtensor::Array{Float64,3}
    Vtensor::Array{Float64,3}
    score::Array{Tuple{Int, Int, Float64},1}  
end


struct PlmOutParallel
    pslike::Union{Vector{Float64},Float64}
    Wtensor::Array{Float64,3}
    Vtensor::Array{Float64,4}
    score::Array{Tuple{Int, Int, Float64},1}  
end

struct PlmVar
    N::Int
    M::Int
    q::Int  
    q2::Int
    H::Int
    lambda::Float64
    Z::Array{Int,2} #MSA
    W::Array{Float64,1} #weigths
end
