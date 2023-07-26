using PrecompileTools

@setup_workload begin
    n_epochs = 1
    batch_size = 10
    d = 2
    H = 2

    const fastafile = "../PlmDCA.jl/data/pf14short.fasta"  
    const structfile = "../ArDCAData/data/PF00014/PF00014_struct.dat" 

    η = 0.005
    n_epoch = 1
    @compile_workload begin
        redirect_stdout(devnull) do
            res1=trainer(fastafile, n_epochs, batch_size = batch_size, H = H, d = 2, structfile = structfile)
        end
    end

end