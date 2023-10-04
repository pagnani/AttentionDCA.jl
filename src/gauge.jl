function gauge_V_ZeroSum(Q,K,V; ar = true)

    H,d,N = size(Q)
    H,q,q = size(V)

    M1 = mean(V, dims=3)[:,:,1]
    V = V .- M1 

    M2 = mean(V, dims=2)[:,1,:]
    for h in 1:H 
        for i in 1:q
            for j in 1:q 
                V[h,i,j] = V[h,i,j] - M2[h,j]
            end
        end
    end
    @tullio sf[i, j, h] := Q[h,d,i]*K[h,d,j]
    sf = softmax_notinplace(sf,dims=2) 
    
    if ar 
        @tullio J[i,j,a,b] := sf[i,j,h]*V[h,a,b]*(i>j)
    else
        @tullio J[i,j,a,b] := sf[i,j,h]*V[h,a,b]*(i!=j)
    end

    F = zeros(q,N)
    for i in 1:N
        for a in 1:q 
            for j in 1:i-1
                @tullio _F = sf[$j,$i,h]*M2[h,$a]
                F[a,i] += _F
            end
            for j in i+1:N 
                @tullio _F = sf[$i,$j,h]*M1[h,$a]
                F[a,i] += _F
            end
        end
    end

    return V, J, F
end
