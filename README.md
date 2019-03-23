# 11747 Team Project
This repo stores the code for 11747 assignment 3 and 4. Our plan is to participate in CoNaLa challenge and obtain state-of-the-art results. In assignment 3, we will reimplement the current SoTA model, TranX. In assignment 4, we will find ways to improve it.

## TranX
The state of the art general-purpose Transition-based abstract syntaX parser that maps natural language queries into machine executable source code (e.g., Python) or logical forms (e.g., lambda calculus).  
Paper: https://arxiv.org/pdf/1810.02720.pdf  
Implementation: https://github.com/pcyin/tranX  

## File Structure
```bash
├── asdl (grammar-based transition system)
├── dataset (dataset specific code like data preprocessing/evaluation/etc.)
├ Model.py (PyTorch implementation of neural nets)
├ conala_main.py (The main function that runs the training and testing procedures.)
```