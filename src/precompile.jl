using PrecompileTools

@setup_workload begin
    n_epochs = 1
    batch_size = 10
    d = 2
    H = 2
    
    const fastafile = "precompilation_data/PF00014.fasta"  
    const structfile = "precompilation_data/PF00014_struct.dat"

    η = 0.005
    n_epoch = 1
    @compile_workload begin
        redirect_stdout(devnull) do
            res1=trainer(fastafile, n_epochs, H = H, d = d, structfile = structfile)
            res2=artrainer(fastafile, n_epochs, H = H, d = 2)
            s = sample(res2[1], 2)
        end
    end

end