module AttentionDCA

import DCAUtils: compute_weighted_frequencies, compute_weights, read_fasta_alignment, remove_duplicate_sequences
import Flux: Adam, DataLoader, gradient
import Flux.Optimise: update!
import Flux.Optimisers: setup

using ArDCA, PottsGauge
using DelimitedFiles: readdlm
using ExtractMacro, Printf
using LinearAlgebra, Tullio, LoopVectorization
using PrecompileTools
using Random



export trainer, stat_trainer, artrainer, stat_artrainer, multi_trainer, multi_artrainer, stat_multi_trainer
export score, compute_freq, compute_PPV, sample, quickread


include("utils.jl")
include("dcascore.jl")
include("attention.jl")
include("multifam.jl")
include("autoregressive.jl")
include("loss_gradient_tests.jl")
end
