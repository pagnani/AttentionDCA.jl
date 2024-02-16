using AttentionDCA, ArDCA
using Test

printstyled("Running tests:\n", bold=true, color=:light_blue)
# Test that the AttentionBasedPlmDCA module is defined.

tests = ["NonGenTest.jl", "GenTest.jl", "MultiFamTest.jl"]

for test in tests
    include("$test")
end

ambiguities=Test.detect_ambiguities(AttentionDCA)
printstyled("Potentially stale exports: ",bold=true, color=:light_blue)
isempty(ambiguities) ? (printstyled("\t none\n",bold=true, color=:light_green)) : (display(ambiguities)) 