using PrecompileTools

@setup_workload begin
    N = 10
    M = 10
    const fastafile = "../PlmDCA.jl/data/pf14short.fasta"  
    const structfile = "../ArDCAData/data/PF00014/PF00014_struct.dat" 
    d = 2
    H = 2
    #Z = rand(1:21,N,M)
    #W = rand(M)
    #W ./= sum(W)

    η = 0.005
    n_epoch = 1
    @compile_workload begin
        redirect_stdout(devnull) do
            Weights, Z, N, M, q = ReadFasta(fastafile, 0.9, :auto, true)
            D = (Z,Weights)
            res1=trainer(D,0.005,M,1, structfile = structfile)
            res2=artrainer(D,0.005,M,1, structfile = structfile)
            res3=attentiondca(fastafile,maxit=2,H=H,d=d)
        end
    end

end