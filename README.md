# AttentionDCA

Package used for the analysis of the factored self-attention mechanism through a simple one-layer DCA model at: 

1. Caredda F., Pagnani A., Direct Coupling Analysis and the Attention Mechanism, [biorxiv:579080](https://www.biorxiv.org/content/10.1101/2024.02.06.579080v1)

## Install

This is an unregistered package: to install enter `]` in the Julia repl and

```
pkg> add https://github.com/pagnani/AttentionDCA.jl

```
## Use

The functions for the training are 
```
trainer, stat_trainer, 
artrainer, stat_artrainer, 
multi_trainer, stat_multi_trainer
multi_artrainer
```
These take as as inputs either tuples with integer-encoded MSA and weight vector $(Z,W)$ or a path to the fasta file containing the sequences of the protein family under study. To get more details on the use of each single function use the help function in the Julia repl, e.g.
```
?trainer
```
## Example 

A detailed example of the use of the package can be found inside `notebooks/ExampleAttentionDCA`. To use it, it is necessary to clone the repository and follow the indications inside the notebook itself.

## Data

All data used in this study is publicly available at [GitHub/francescocaredda/DataAttentionDCA](https://github.com/francescocaredda/DataAttentionDCA) in the "data" folder.


## Inquiries 

Any question can be directed to francesco.caredda@polito.it