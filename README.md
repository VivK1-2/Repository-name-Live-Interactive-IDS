# OST Live Interactive IDS Lab

This version changes the project from a static results viewer into a live experiment workflow.

## Workflow

1. Run `streamlit run app.py`
2. Open **Train Model**
3. Upload a labelled training CSV.
4. Train Random Forest + a differentiable PyTorch IDS.
5. Open **Traffic Analyzer**
6. Upload unseen network traffic CSV.
7. The application predicts Normal vs Attack and reports:
   - total records
   - detected attacks
   - attack rate
   - row-level predictions
   - downloadable prediction CSV
8. Open **Adversarial Lab**
9. Select FGSM or PGD and run an evasion experiment against the PyTorch IDS.

## UNSW-NB15

For the binary experiment, use `label` as the target:
- 0 = Normal
- 1 = Attack

The app automatically handles numeric/categorical columns using imputation, one-hot encoding and standardization.

## Important

The adversarial lab performs model-feature-space robustness testing. It does NOT generate real malicious packets or guarantee that perturbed features correspond to valid network traffic.

## Windows / VS Code

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again.

## Recommended demo

Use the original UNSW-NB15 training CSV to train the model, then upload the separate test CSV to Traffic Analyzer. For a clean demonstration, keep a copy of the untouched test CSV and compare the predicted attack rate before running the adversarial lab.

## License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for the complete license text.
