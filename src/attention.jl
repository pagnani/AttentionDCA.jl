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

function attention_plsiteandgrad!(x::Vector{}, grad::Vector{}, site::Int, plmvar::PlmVar)
    
    pgrad = pointer(grad)

    N = plmvar.N
    M = plmvar.M
    q = plmvar.q
    Z = plmvar.Z

    H = plmvar.H 
    
    L = H*N*N + H*q*q 

    L == length(x) || error("Wrong dimension of parameter vector")
    L == length(grad) || error("Wrong dimension of gradient vector")


    W_site = reshape(x[1:H*N*N],H,N,N)[:,site,:]
    V = reshape(x[H*N*N+1:end],H,q,q)


    Wsf_site = softmax(W_site,dims=2)
    @tullio J[a,b,j] := Wsf_site[h,j]*V[h,a,b]*(site!=j)
    
    @tullio mat_ene[a,m] := J[a,Z[j,m],j]

    pseudologlikelihood = 0.0 
    partition = sumexp(mat_ene,dims=1)
    @tullio prob[a,m] := exp(mat_ene[a,m])/partition[m]
    lge = log.(partition) 
    Z_site = view(Z,site,:)
    @tullio pseudologlikelihood = mat_ene[Z_site[m],m] - lge[m]
    pseudologlikelihood /= -M


    grad .= 0.0
    @tullio mat[j,a,b] := (Z[j,m]==b)*((Z_site[m]==a)-prob[a,m]) (a in 1:q, b in 1:q)
    mat[site,:,:] .= 0.0

    @inbounds for counter = 1:N*N*H
        h,r,i = counter_to_index(counter,N,q,H)
        if r==site
            for j = 1:N
                for a = 1:q 
                    @simd for b = 1:q 
                        grad[counter] += mat[j,a,b]*V[h,a,b]*((i==j)*Wsf_site[h,i]-Wsf_site[h,i]*Wsf_site[h,j])      
                    end
                end
            end 
        end
        grad[counter] /= -M
    end

    for counter = N*N*H+1:L
        h,c,d = counter_to_index(counter,N,q,H)
        for j = 1:N
            grad[counter]+=Wsf_site[h,j]*mat[j,c,d]
        end        
        grad[counter] /= -M 
    end
    pointer(grad)==pgrad || error("Different pointer")
    return pseudologlikelihood
end

function logsumexp(a::AbstractArray{<:Real}; dims=1)
    m = maximum(a; dims=dims)
    return m + log.(sum(exp.(a .- m); dims=dims))
end

function sumexp(a::AbstractArray{<:Real};dims=1)
    m = maximum(a; dims=dims)
    return sum(exp.(a .- m ).*exp.(m); dims=dims)
end