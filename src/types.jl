struct PlmAlg
    method::Symbol
    verbose::Bool
    epsconv::Float64
    maxit::Int
end

struct AttPlmVar
    N::Int
    M::Int
    d::Int
    dd::Float64
    q::Int  
    q2::Int
    H::Int
    lambda::Float64
    Z::Array{Int,2} #MSA
    W::Array{Float64,1} #weigths
    delta::Array{Int,3}
    wdelta::Array{Float64,3}
    function AttPlmVar(N,M,d,q,H,lambda,Z,Weigths; dd = Float64(d))
        
        @tullio delta[j,m,a] := Int(Z[j,m]==a) (a in 1:q)
        @tullio wdelta[j,m,a] := Weigths[m]*delta[j,m,a]

        new(N,M,d,dd,q,q*q,H,lambda,Z,Weigths,delta,wdelta)
    end
end

function Base.show(io::IO, AttPlmVar::AttPlmVar)
    @extract AttPlmVar: N M d q H lambda
    print(io,"AttPlmVar: \nN=$N\nM=$M\nq=$q\nH=$H\nd=$d\nλ=$(lambda)")
end

struct FieldAttPlmVar
    N::Int
    M::Int
    d::Int
    dd::Float64
    q::Int  
    q2::Int
    H::Int
    lambdaJ::Float64
    lambdaF::Float64
    Z::Array{Int,2} #MSA
    W::Array{Float64,1} #weigths
    delta::Array{Int,3}
    wdelta::Array{Float64,3}
    function FieldAttPlmVar(N,M,d,q,H,lambdaJ,lambdaF,Z,Weigths; dd = Float64(d))
        
        @tullio delta[j,m,a] := Int(Z[j,m]==a) (a in 1:q)
        @tullio wdelta[j,m,a] := Weigths[m]*delta[j,m,a]

        new(N,M,d,dd,q,q*q,H,lambdaJ,lambdaF,Z,Weigths,delta,wdelta)
    end
end

function Base.show(io::IO, FieldAttPlmVar::FieldAttPlmVar)
    @extract FieldAttPlmVar: N M d q H lambdaJ lambdaF
    print(io,"FieldAttPlmVar: \nN=$N\nM=$M\nq=$q\nH=$H\nd=$d\nλJ=$(lambdaJ)\nλH=$(lambdaF)")
end

# struct AttPlmOut
#     Q::Array{Float64,3}
#     K::Array{Float64,3}
#     V::Array{Float64,3}
#     pslike::Union{Vector{Float64},Float64}
# end

# function Base.show(io::IO, AttPlmOut::AttPlmOut)
#     @extract AttPlmOut: Q K V pslike
#     H,d,N = size(Q)
#     H,q,q = size(V) 
#     print(io,"AttPlmOut: \nsize(Q)=[$H,$d,$N]\nsize(K)=[$H,$d,$N]\nsize(V)=[$H,$q,$q]\npslike=$(pslike)")
# end

# struct FieldAttPlmOut
#     Q::Array{Float64,3}
#     K::Array{Float64,3}
#     V::Array{Float64,3}
#     F::Array{Float64,2}
#     pslike::Union{Vector{Float64},Float64}
# end

struct AttOut
    Q::Array{Float64,3}
    K::Array{Float64,3}
    V::Array{Float64,3}
    F::Union{Array{Float64,2},Nothing}
    pslike::Union{Vector{Float64},Float64}
end

function Base.show(io::IO, AttOut::AttOut)
    @extract AttOut: Q K V F pslike
    H,d,N = size(Q)
    H,q,q = size(V) 
    if F === nothing 
        print(io,"AttOut: \nsize(Q)=[$H,$d,$N]\nsize(K)=[$H,$d,$N]\nsize(V)=[$H,$q,$q]\npslike=$(pslike)\nNOFIELDS")
    else       
        print(io,"AttOut: \nsize(Q)=[$H,$d,$N]\nsize(K)=[$H,$d,$N]\nsize(V)=[$H,$q,$q]\nsize(F)=[$q,$N]\npslike=$(pslike)\nFIELDS")
    end
end

# struct OldAttComputationQuantities 
#     sf::Array{Float64,3}
#     J::Array{Float64,4}
#     mat::Array{Float64,4}
#     fact::Array{Float64,3}
#     function OldAttComputationQuantities(N,H,q)
#         sf = zeros(Float64, H, N, N)
#         J = zeros(Float64, N, N, q, q)
#         mat = zeros(Float64, N, q, q, N)
#         fact = zeros(Float64, N, H, N)
#         new(sf,J,mat,fact)
#     end
# end

struct AttComputationQuantities 
    sf::Array{Float64,3}
    J::Array{Float64,4}
    mat::Array{Float64,4}
    function AttComputationQuantities(N,H,q)
        sf = zeros(Float64, N, N, H)
        J = zeros(Float64, N, N, q, q)
        mat = zeros(Float64, N, q, q, N)
        new(sf,J,mat)
    end
end


struct myAttPlmVar
    N::Int
    M::Int
    d::Int
    dd::Float64
    q::Int  
    q2::Int
    H::Int
    lambdaQ::Float64
    lambdaV::Float64
    Z::Array{Int,2} #MSA
    W::Array{Float64,1} #weigths
    delta::Array{Int,3}
    wdelta::Array{Float64,3}
    function myAttPlmVar(N,M,d,q,H,lambdaQ,lambdaV,Z,Weigths; dd = Float64(d))
        
        @tullio delta[j,m,a] := Int(Z[j,m]==a) (a in 1:q)
        @tullio wdelta[j,m,a] := Weigths[m]*delta[j,m,a]

        new(N,M,d,dd,q,q*q,H,lambdaQ,lambdaV,Z,Weigths,delta,wdelta)
    end
end

struct myFieldAttPlmVar
    N::Int
    M::Int
    d::Int
    dd::Float64
    q::Int  
    q2::Int
    H::Int
    lambdaQ::Float64
    lambdaV::Float64
    lambdaF::Float64
    Z::Array{Int,2} #MSA
    W::Array{Float64,1} #weigths
    delta::Array{Int,3}
    wdelta::Array{Float64,3}
    function myFieldAttPlmVar(N,M,d,q,H,lambdaJ,lambdaV,lambdaF,Z,Weigths; dd = Float64(d))
        
        @tullio delta[j,m,a] := Int(Z[j,m]==a) (a in 1:q)
        @tullio wdelta[j,m,a] := Weigths[m]*delta[j,m,a]

        new(N,M,d,dd,q,q*q,H,lambdaJ,lambdaV,lambdaF,Z,Weigths,delta,wdelta)
    end
end

struct Info
    λQ::Union{Nothing, Float64} 
    λV::Union{Nothing, Float64}
    λJ::Union{Nothing, Float64}
    λF::Union{Nothing, Float64}
    numevals::Int 
    elapstime::Float64
    ret::Symbol
    function Info(λQ,λV,λF,numevals,elapstime,ret)
        new(λQ,λV,nothing,λF,numevals,elapstime,ret)
    end
    function Info(λJ,λF,numevals,elapstime,ret)
        new(nothing,nothing,λJ,λF,numevals,elapstime,ret)
    end
end

function Base.show(io::IO, Info::Info)
    @extract Info: λQ λV λJ λF numevals elapstime ret
    println("\nInfo:")
    if λQ === nothing 
        println("λJ = $λJ")
    else
        println("λQ = $λQ")
        println("λV = $λV")
    end
    if λF !== nothing 
        println("λF = $λF")
    end
    println("Number of evaluation = $numevals")
    println("Elapsed time = $elapstime")
    println("Stopping criterion = $ret")
end