# medtok_reimplement

This is an independent re-implementation of the paper: <br/>
"Multimodal Medical Code Tokenizer" <br/>
Authors: Xiaorui Su, Shvat Messica, Yepeng Huang, Ruth Johnson, Lukas Fesser, Shanghua Gao, Faryad Sahneh, Marinka Zitnik <br/>
Published: ICML '25 <br/>
Link to official code: https://github.com/mims-harvard/MedTok <br/>

## Purpose
This project aims to reproduce and understand the core methods proposed in the paper. The authors of the original paper introduce a tokenizer built specifically for medical purposes which is not only familiar with medical vocabulary, but is also learned about the nuances and hierarchies between terms.  <br/>
**The code in this repository is not affiliated with or derived from the official implementation.**

## Key Features (in progress)
- [x] Written in PyTorch
- [x] Implements the main model described in the paper
- [ ] Implements the training loop described in the paper
- [ ] Implements preprocessing methods to generate graphs as described in the paper
- [ ] Provide weights
- [x] Educational and open for community improvements

## Disclaimer
This is an unofficial re-implementation for educational and research purposes, and to further my own understanding of the methods described in the paper. This codebase is part of a series of independent implementations of recent ML papers to build practical understanding and demonstrate reproducibility.  <br/>
**Most importantly, all credits for the original idea and research go to the authors of the paper.**
