using Revise
using AttentionDCA, PlmDCA
using Statistics, LinearAlgebra, Tullio
using PyPlot, JLD2, DataFrames, GLM

includet("useful_functions.jl")

function graphPPV(PPVs, labels, figtitle::String;
    fig_size=(8,6),
    colors = ["r","b","g","y"],
    fs = 15)
    
    close("all")
    fig = figure(figsize=fig_size)
    
    l = length(PPVs)
    if l>1
    #The first PPV is the structure, the second is from PlmDCA
        semilogx(PPVs[1], ".-", ms = 1,label = "Structure", c = "gray")
        semilogx(PPVs[2], ".-", label = labels[1], c = "k")
        [semilogx(PPVs[i+2], ".-", label = labels[i+1], c = colors[i]) for i in 1:length(PPVs)-2]
    else
        semilogx(PPVs[1], ".-", ms = 1)
    end
    
    xticks(fontsize=fs)
    yticks(fontsize=fs)
    legend(fontsize=fs)
    xlabel("Number of Predictions", fontsize=fs)
    ylabel("PPV", fontsize=fs)
    title(figtitle, fontsize=fs)
    
end
    

function graphAtt(Q, K, V, filestruct, PFname, ticks; k = nothing, version = mean, sqr = false, APC = true)
    
    ms = 100/size(Q,3)
    
    Am = if k === nothing k_matrix(Q,K,V, version; sym = true, APC = APC, sqr = sqr)[1]
    else
        k_matrix(Q,K,V, k, version; sym = true, APC = APC, sqr = sqr)[1]
    end
    
    #this is to fill the diagonal and have a nicer plot
    @tullio Am[i,i] = mean(Am)
    
    imshow(Am, cmap = "gray_r")
    
    dist = true_structure(filestruct)
    
    plot(dist[:,1].-1, dist[:,2].-1, c="r", "o", alpha=0.1, ms=ms, label = "True Structure")
    plot(dist[:,2].-1, dist[:,1].-1, c="r", "o", alpha=0.1, ms=ms)
    
    xticks(ticks.-1,ticks,fontsize=20)
    yticks(ticks.-1,ticks,fontsize=20)
    title(PFname*", Attention Map", fontsize = 20)
    colorbar(shrink = 0.7, aspect = 10)
    
end    

function contact_plot(score,filestruct,L,figurename;ticks = nothing, min_separation=0, cutoff=8.0, N = "2L")
    
    dist = AttentionDCA.compute_residue_pair_dist(filestruct)
    roc = map(x->x[4],AttentionDCA.compute_referencescore(score, dist, cutoff=cutoff, mindist=min_separation))
    precision = round(roc[L], digits=2)
    for (key,value) in dist
        if key[2]-key[1]<=min_separation || value > cutoff || value == 0 
            delete!(dist,key)
        end
    end
    # predicted_contacts = [(score[i][1],score[i][2]) for i in 1:L if score[i][2]-score[i][1]>min_separation]
    predicted_contacts = []
    i = 1
    while length(predicted_contacts) < L
        if i>length(score)
            break
        end
        if score[i][2]-score[i][1]>min_separation
            push!(predicted_contacts,(score[i][1],score[i][2]))
        end
        i += 1
    end

    true_contacts = []
    false_contacts = []
    for contact in predicted_contacts
        if haskey(dist,contact)
            push!(true_contacts,contact)
            delete!(dist,contact)
        else
            push!(false_contacts,contact)
        end
    end
    true_contacts = reduce(hcat, getindex.(true_contacts,i) for i in 1:2)
    false_contacts = reduce(hcat, getindex.(false_contacts,i) for i in 1:2)
    dist = reduce(hcat,getindex.(collect(keys(dist)),i) for i in 1:2)
       
    close("all")
    #fig_size= (10,8)
    plot(dist[:,1],dist[:,2],c="k","o",alpha=0.1,ms=2,markeredgewidth=2)
    plot(dist[:,2],dist[:,1],c="k","o",alpha=0.1,ms=2,markeredgewidth=2)
            
    plot(true_contacts[:,1],true_contacts[:,2],c="b","o",alpha=0.5,ms=2,markeredgewidth=2)
    plot(true_contacts[:,2],true_contacts[:,1],c="b","o",alpha=0.5,ms=2,markeredgewidth=2)
           
    plot(false_contacts[:,2],false_contacts[:,1],c="r","o",alpha=0.5,ms=2,markeredgewidth=2)
    plot(false_contacts[:,1],false_contacts[:,2],c="r","o",alpha=0.5,ms=2,markeredgewidth=2)
    
            
    xticks(ticks,fontsize=20)
    yticks(ticks,fontsize=20)
            
    #xlabel("i", fontsize=20)
    #ylabel("j", fontsize=20)
    
    #xlim((0,maximum(dist)+1))
    #ylim((maximum(dist)+1,0))
    xlim((1,maximum(dist)))
    ylim((maximum(dist),1))
    tight_layout(pad=1.0)
    axis("scaled")
    title(figurename*" Contact Map, P@"*N*": $precision", fontsize=20)

       
    #return dist, true_contacts, false_contacts, ticks

end



function graphConnCorr(ar, ar_ref, nsample, name::String; fs = 25)
    #fig = figure(figsize=(12, 10))
    f1,f2 = compute_freq(ar[2].Z, ar[2].W)
    c = f2 - f1*f1'
    
    Zs = sample(ar[1], nsample)
    f1s,f2s = compute_freq(Zs)
    cs = f2s - f1s*f1s'
    
    model = lm(@formula(y ~ x), DataFrame(x=c[:], y=cs[:]))
    m = coef(model)[2]
    q = coef(model)[1]
    y_pred = m * c[:] .+ q
    rmse = sqrt(mean((cs[:] .- y_pred).^2))

    
    Zss = sample(ar_ref[1], nsample)
    f1ss,f2ss = compute_freq(Zss)
    css = f2ss - f1ss*f1ss'
    
    
    model = lm(@formula(y ~ x), DataFrame(x=c[:], y=css[:]))
    m_ardca = coef(model)[2]
    q_ardca = coef(model)[1]
    y_pred = m_ardca * c[:] .+ q_ardca
    rmse_ardca = sqrt(mean((css[:] .- y_pred).^2))
    
    
    fig = figure(figsize=(8,6.5))
    
    plot(c[:], css[:], ".", c = "r", label = "ArDCA \npearson = $(round(cor(c[:],css[:]), digits = 3)), rms = $(round(rmse_ardca, digits = 4))")
    plot(range(minimum(c[:]), maximum(c[:]), 2), m_ardca.*range(minimum(c), maximum(c), 2) .+ q_ardca, "--", c = "tab:red")
    
    
    plot(c[:], cs[:], ".", label = "AttentionDCA \npearson = $(round(cor(c[:],cs[:]), digits = 3)), rms = $(round(rmse, digits = 4))")
    plot(range(minimum(c[:]), maximum(c[:]), 2), m.*range(minimum(c), maximum(c), 2) .+ q, "--", c = "tab:blue")
    
    xticks(fontsize=fs)
    yticks(fontsize=fs)
    
    
    xlabel("Natural", fontsize=fs)
    ylabel("Generated", fontsize=fs)
    
    
    title(name*", Connected 2-Site Correlations", fontsize=fs)
    
    Zss = sample(ar_ref[1], nsample)
    f1ss,f2ss = compute_freq(Zss)
    css = f2ss - f1ss*f1ss'
    
    
    model = lm(@formula(y ~ x), DataFrame(x=c[:], y=css[:]))
    m_ardca = coef(model)[2]
    q_ardca = coef(model)[1]
    y_pred = m_ardca * c[:] .+ q_ardca
    rmse_ardca = sqrt(mean((css[:] .- y_pred).^2))
     
    #ax1 = fig.add_axes([1.70,0.37,1.25,1.25])
    #plot(c[:], css[:], ".", c = "r", label = "ArDCA \ncorr = $(round(cor(c[:],css[:]), digits = 3)), σ = $(round(rmse_ardca, digits = 4))")
    #plot(range(minimum(c[:]), maximum(c[:]), 2), m_ardca.*range(minimum(c), maximum(c), 2) .+ q_ardca, "--", c = "tab:red")
    #title("ArDCA")
    legend(loc = 4, fontsize = 16, framealpha=0.0)
    #legend(loc=2, markerscale = 10, fontsize = fs-10)
    
    #return c, cs, css, m, q, rmse, m_ardca, q_ardca, rmse_ardca
end