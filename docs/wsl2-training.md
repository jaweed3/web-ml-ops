# WSL2 Full Training Setup (PC Lab — Windows + RTX 4060)

This guide sets up WSL2 on the PC Lab so you can run full CUDA training with one command.

---

## 1. Install WSL2

Open **PowerShell as Administrator**:

```powershell
wsl --install
```

This installs WSL2 + Ubuntu 24.04 by default. Reboot when prompted.

After reboot, Ubuntu will launch and ask for a username/password. Set anything — this is your local Linux user.

Verify CUDA is visible inside WSL2:

```bash
nvidia-smi
```

If you see your RTX 4060, you're good. CUDA drivers from Windows are automatically passed through to WSL2 — no separate Linux CUDA install needed.

---

## 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

---

## 3. Clone the repo

```bash
git clone git@github.com:jaweed3/web-ml-ops.git
cd web-ml-ops
```

If SSH key isn't set up in WSL2 yet:

```bash
ssh-keygen -t ed25519 -C "pclab"
cat ~/.ssh/id_ed25519.pub
# Add this key to GitHub → Settings → SSH Keys
```

---

## 4. Install training dependencies

```bash
uv sync --all-groups
```

This installs ultralytics + tensorflow + all training deps. Takes a few minutes first time, then cached.

---

## 5. Configure environment

Create `.env` in the repo root:

```bash
cat > .env << 'EOF'
DAGSHUB_USERNAME=jaweed3
DAGSHUB_REPO=web-ml-ops
DAGSHUB_TOKEN=your_dagshub_token_here
EOF
```

Load it:

```bash
export $(cat .env | xargs)
```

---

## 6. Configure DVC credentials

```bash
dvc remote modify storage --local access_key_id $DAGSHUB_TOKEN
dvc remote modify storage --local secret_access_key $DAGSHUB_TOKEN
```

This writes credentials to `.dvc/config.local` (gitignored).

---

## 7. Run full training — one command

```bash
make all
```

This runs all 5 stages in sequence:
1. `stage1_data` — pull full dataset from DagHub via DVC
2. `stage2_train` — 50 epochs, CUDA, full dataset
3. `stage3_export` — ONNX fp32 + int8 + TFLite
4. `stage4_benchmark` — latency + mAP evaluation
5. `stage5_register` — register to MLflow Staging (metric gate enforced)

Then manually trigger promotion from GitHub Actions (you'll get Telegram notification).

---

## 8. Push model to DagHub after training

After `make all` completes, the model is registered to **Staging** in MLflow.

Go to GitHub → Actions → Run workflow → set `promote_to_production = true` to promote to Production.

Or wait — on every push to main from any machine, CI auto-promotes after your Telegram approval.

---

## Everyday workflow (after initial setup)

```bash
cd web-ml-ops
git pull
export $(cat .env | xargs)
make all
```

That's it. One command, full training, auto-register to Staging.

---

## Troubleshooting

**`nvidia-smi` not found in WSL2**
- Make sure Windows NVIDIA drivers are up to date (≥ 525.x)
- Don't install CUDA inside WSL2 — it's inherited from Windows

**`dvc pull` 401 error**
- Token expired — generate new one at dagshub.com/user/settings/tokens
- Re-run step 6

**Out of memory during training**
- Reduce batch size: edit `configs/train_config.yaml` → `batch: 8`
- RTX 4060 8GB should handle batch 16 fine at 640px
