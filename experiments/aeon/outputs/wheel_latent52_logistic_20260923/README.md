# Aeon Figure 5 single-latent decoder

Latent 52 (D=192, k=12), wheel-rim speed >0.75 cm/s.

Test AUROC: **0.926085312**. Validation AUROC: 0.902052735. Selected C: 0.001.

Chronological train/validation/test intervals are the first 14.4 hours, next 1.8 hours, and final 1.8 hours. Input windows crossing these boundaries are excluded. Standardization and logistic coefficients are fitted on training data only; regularization is selected on validation AUROC without refitting.

Test support: 323,860 windows, including 7,892 positives.

Exploratory/transductive: frozen neural model, normalization, latent and feature were previously fitted or selected using the full recording.
