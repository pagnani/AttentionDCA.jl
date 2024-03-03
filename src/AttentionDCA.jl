module AttentionDCA

import CUDA
import cuDNN
import CUDA:CuArray
import DCAUtils: compute_weighted_frequencies, compute_weights, read_fasta_alignment, remove_duplicate_sequences
import Flux
import Flux: Adam, DataLoader, gradient, softmax, cpu, gpu
import Flux.Optimise: update!
import Flux.Optimisers: setup
import NNlib
import NNlib:dot_product_attention,batched_mul
using KernelAbstractions

#using LoopVectorization
using Tullio 
using ArDCA, PottsGauge
using DelimitedFiles: readdlm
using ExtractMacro, Printf
using LinearAlgebra
using PrecompileTools
using Random, Statistics




export trainer, stat_trainer, artrainer, stat_artrainer, multi_trainer, multi_artrainer, stat_multi_trainer
export score, compute_freq, compute_PPV, sample, quickread


include("utils.jl")
include("dcascore.jl")
include("attention.jl")
include("multifam.jl")
include("autoregressive.jl")
include("andrea_KL.jl")
#include("precompile.jl")
#include("embedding.jl")
include("embedding_KL.jl")
end
