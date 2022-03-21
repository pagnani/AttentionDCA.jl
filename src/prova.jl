function my_plsiteandgrad(x::Vector{Float64}, grad::Vector{Float64},att::Attention, plmvar::PlmVar, site)
    # x è un vettore di N*N*H + Q*Q*H parametri che corrispondono ai parametri dell'attention 
    N::Int = att.N
    H::Int = att.H
    q::Int = att.q
    M::Int = plmvar.M 
    J = att.J
    Wsf = att.Wsf


    pseudolike = 0.0
    vecene = zeros(Float64, q)
    expvecenesumnorm = zeros(Float64, q)

    for i = 1:M 
        my_fillvecene!(vecene, plmvar, att, site, m)
        log_m = logsumexp(vecene)
        pseudolike -= log_m 
        for i = 1:site-1 
            pseudolike += J[Z[site,m],Z[i,m],site,i]
        end
        for i = site+1:N 
        pseudolike += J[Z[site,m],Z[i,m],site,i]
        end
    end 
    pseudolike *= - 1/M


    LL = N*N*H + q*q*H

    for x = 1:N*N*H
        scra = 0.0 
        h, site, y = counter_to_index(x, N, q, H)   
        for i = 1:N 
            for a = 1:q 
                for b = 1:q
                    scra += computegrad_W(plmvar, site, h, i, y, a, b, att)
                end
            end
        end
        grad[x] = scra 
    end       

    for x = N*N*H+1:LL 
        scra = 0.0 
        h, c, d = counter_to_index(x, N, q, H)
        for i = 1:N 
            scra += computegrad_V(plmvar, att, site, i, c, d, h)  
        end
        grad[x] = scra
    end
end