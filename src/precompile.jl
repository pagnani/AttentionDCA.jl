using PrecompileTools

@setup_workload begin
    N = 10
    M = 10
    #d = 2
    #H = 2
    Z = rand(1:21,N,M)
    W = rand(M)
    W ./= sum(W)
    @compile_workload begin
        redirect_stdout(devnull) do
            attentiondca(Z,W)
        end
    end
end