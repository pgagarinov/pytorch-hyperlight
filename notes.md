1. For some reason f1 = accuracy
2. ASHA scheduler doesn't user the best observed score (the one that EarlyStopping callback uses for early stopping) for stopping
3. We need CLILogger to have a persistent console output (in addition to the progress bar or instead of it)