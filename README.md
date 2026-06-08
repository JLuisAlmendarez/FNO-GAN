The following has on purpose to present the organization of the github repository: 
```
.
├── Analysis
│   ├── Initial             # Exploratory analysis of raw dataset and physical patterns
│   │   ├── Data Analysis (Raw Dataset).ipynb
│   │   └── Dataset Election Analysis.ipynb
│   └── Models (if)         # (reserved) analysis of trained model behaviors
├── lib                     # Reusable core modules for all training approaches
│   ├── Common.py           # Dataset, FNO blocks, generator, NS residual, BaseTrainer, spectral tools
│   ├── GenAdvNetworkApproach.py   # WGAN‑GP trainer with dual discriminators (statistical & physical)
│   ├── Lebesgue2Approach.py       # Supervised trainer – minimizes one‑step MSE
│   ├── PurePhysLossApproach.py    # Physics‑informed trainer – minimizes PDE residual
│   └── __pycache__/               # Compiled bytecode (auto‑generated)
├── Models
│   ├── Dataset             # Simulation data (raw + preprocessed)
│   │   ├── Kolmogorov2d_fp32_64x64_N1152_Re1000_T100.pt   # Original dataset from CFD solver
│   │   └── snapshots_64x64_use.npy                        # Normalized vorticity snapshots ready for training
│   ├── Neural_Networks     # Notebooks to launch each learning strategy
│   │   ├── FNO_Lebesgue2.ipynb       # L2‑supervised FNO training
│   │   ├── FNO_ResPhyLoss.ipynb     # Physics‑loss FNO training
│   │   ├── GA_FNO_Prob.ipynb        # WGAN‑GP FNO training
│   │   ├── *.log                     # Log files capturing console output per experiment
│   │   └── logs_*/                   # Checkpoints (latest & best model) and training history (JSON)
│   ├── Preprocessing Pipeline   # Data preparation from raw .pt to .npy
│   │   └── Preprocess Pipe.ipynb
│   └── Tests                 # Quick experiments and model sanity checks
│       └── Experiments.ipynb
├── README.md
├── Report                  # Scientific paper source and compiled document
│   ├── llncs/               # Springer LNCS template files
│   ├── new.*                # Main LaTeX manuscript (source, auxiliary, PDF)
│   ├── references/          # Bibliography files
│   └── respaldo/            # Backups of previous paper drafts
└── Visz                    # Figures and animations used in the report
    ├── ibm_zeroshot.gif
    ├── super_resolution.gif
    └── zeroshot_domain_decomp.gif

```

