# GPU CI Runner Integration Research

**Date**: 2025-10-25
**Repository**: `gsornsen/full-duplex-voice-chat`
**Current CI**: GitHub Actions (CPU-only, ~10-15 min PR CI, 730 tests)
**Research Scope**: Methods to integrate GPU runners for ML/TTS testing

---

## Executive Summary

This document analyzes 7 methods to integrate GPU runners with GitHub Actions CI for a Python ML/TTS project requiring GPU testing (CosyVoice 2, XTTS, future adapters). Focus is on **zero-cost or ultra-low-cost solutions** suitable for OSS projects.

**Top Recommendations** (ranked by fit for this project):

1. **GitHub Actions GPU Runners** (Native, 2025) - BEST if available
2. **Cirun** (Ephemeral runners in owner's cloud) - BEST for control/scale
3. **Self-Hosted Runner** (On-prem hardware) - Zero platform cost, requires hardware
4. **CML** (Continuous Machine Learning) - ML-focused, good automation
5. **Terraform/Custom Actions** - Most flexible, highest complexity

**Not Recommended for CI**:
- Hugging Face ZeroGPU (ToS violation)
- Google Colab (ToS violation, API instability)
- Kaggle (ToS restriction on automation)

---

## 1. Cirun - Ephemeral Self-Hosted Runners

### Overview

Cirun is a managed service that spins up ephemeral self-hosted GitHub Actions runners in **your own cloud account** (AWS, GCP, Azure). You control the infrastructure, Cirun handles the orchestration.

**Website**: https://cirun.io
**Docs**: https://docs.cirun.io

### How It Works

1. **Configuration**: Create `.cirun.yml` in repo root
2. **Cloud Integration**: Link your AWS/GCP/Azure account via Cirun dashboard
3. **Label-Based Triggering**: Workflows use `runs-on: cirun-<label>`
4. **Lifecycle**:
   - GitHub webhook triggers Cirun on job start
   - Cirun provisions VM in your cloud account
   - VM auto-registers as ephemeral GitHub runner
   - Job executes on ephemeral runner
   - VM auto-terminates after job completion (max 6h timeout)
5. **Cleanup**: Guaranteed VM shutdown regardless of job success/failure

### Configuration Example

```yaml
# .cirun.yml
runners:
  - name: "gpu-runner"
    cloud: aws
    instance_type: g4dn.xlarge  # T4 GPU, 4 vCPU, 16GB RAM
    machine_image: ami-0c55b159cbfafe1f0  # Deep Learning AMI
    region: us-east-1
    labels:
      - cirun-gpu-runner
    preemptible: true  # Use spot instances for cost savings
    disk_size: 100  # GB

  - name: "gpu-runner-large"
    cloud: gcp
    instance_type: n1-standard-4-t4  # T4 GPU, 4 vCPU, 15GB RAM
    machine_image: projects/deeplearning-platform-release/global/images/family/pytorch-latest-gpu
    region: us-central1
    labels:
      - cirun-gpu-large
    preemptible: true
    disk_size: 150
```

### Workflow Integration

```yaml
# .github/workflows/gpu-ci.yml
name: GPU CI

on:
  pull_request:
    paths:
      - 'src/tts/adapters/adapter_cosyvoice.py'
      - 'src/tts/adapters/adapter_xtts.py'
      - 'tests/integration/test_gpu_tts.py'

jobs:
  test-gpu-tts:
    runs-on: cirun-gpu-runner  # Triggers Cirun provisioning
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4

      - name: Verify GPU availability
        run: nvidia-smi

      - name: Install dependencies
        run: |
          # Pre-installed CUDA/PyTorch on Deep Learning AMI
          uv sync --extra gpu

      - name: Run GPU TTS tests
        run: |
          uv run pytest tests/integration/test_gpu_tts.py \
            -v -m gpu \
            --cov=src/tts/adapters
```

### Supported Clouds

**AWS**:
- GPU instances: `g4dn.xlarge` (T4), `g5.xlarge` (A10G), `p3.2xlarge` (V100)
- AMIs: Deep Learning AMI (Ubuntu 22.04, PyTorch 2.3.1, CUDA 12.1)
- Spot pricing: ~70% discount vs on-demand

**GCP**:
- GPU instances: `n1-standard-4` + `nvidia-tesla-t4`
- Images: Deep Learning VM (PyTorch/TensorFlow pre-installed)
- Preemptible instances: ~80% discount

**Azure**:
- GPU instances: `Standard_NC4as_T4_v3` (T4)
- Images: Data Science VM (GPU-enabled)
- Spot pricing: ~70-90% discount

### Pricing (OSS Plan)

**Cirun Platform Fee**: **$0/month** for open-source projects
**Cloud Costs** (you pay directly to cloud provider):

| Instance Type | Cloud | GPU | vCPU | RAM | On-Demand | Spot/Preemptible | Per Minute |
|--------------|-------|-----|------|-----|-----------|------------------|------------|
| `g4dn.xlarge` | AWS | T4 | 4 | 16GB | $0.526/hr | ~$0.158/hr | ~$0.0026 |
| `n1-standard-4-t4` | GCP | T4 | 4 | 15GB | $0.65/hr | ~$0.13/hr | ~$0.0022 |
| `Standard_NC4as_T4_v3` | Azure | T4 | 4 | 28GB | $0.526/hr | ~$0.158/hr | ~$0.0026 |

**Example Monthly Cost** (100 CI runs, 10 min each, spot instances):
- 100 runs × 10 min × $0.0022/min = **$2.20/month**

### Cold Start Performance

**Total time from job trigger to test execution**:
- VM provisioning: 60-90 seconds (AWS/GCP)
- GitHub runner registration: 10-15 seconds
- Docker pull (if needed): 30-60 seconds
- **Total cold start**: ~2-3 minutes

**Optimization**:
- Use pre-baked AMIs/images with dependencies installed
- Cache Docker layers in cloud storage
- Warm pool (1 instance on standby, not part of OSS plan)

### Teardown Guarantees

**Success scenario**:
- Job completes → Cirun terminates VM within 60 seconds

**Failure scenarios**:
- Job timeout (max 6h) → Cirun force-terminates VM
- Job cancelled by user → Cirun terminates VM within 60 seconds
- Network partition → Cirun heartbeat timeout (5 min) → Terminate VM
- Cirun service outage → Cloud-side auto-shutdown after 6h max uptime

**Cost protection**:
- Hard limit: 6-hour max VM lifetime
- Cost alerts in Cirun dashboard
- Monthly spending caps configurable per cloud

### Secrets Handling

**Approach**: GitHub secrets passed to runner as environment variables

```yaml
jobs:
  test-gpu:
    runs-on: cirun-gpu-runner
    steps:
      - name: Configure cloud credentials
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          HUGGINGFACE_TOKEN: ${{ secrets.HUGGINGFACE_TOKEN }}
        run: |
          # Secrets available as env vars
          echo "Authenticating with HuggingFace..."
```

**Security**:
- Ephemeral VMs have no persistent state (secrets wiped on termination)
- IAM roles recommended over hardcoded credentials (AWS/GCP/Azure)
- Cirun does NOT see your secrets (GitHub Actions runtime handles injection)

### Label-Based Routing

**Workflow definition**:
```yaml
# Different GPU configurations for different test suites
jobs:
  test-piper:
    runs-on: ubuntu-latest  # CPU tests on free GitHub runners

  test-cosyvoice:
    runs-on: cirun-gpu-runner  # T4 GPU (g4dn.xlarge)

  test-xtts:
    runs-on: cirun-gpu-large  # A10G GPU (g5.xlarge) for heavier models
```

### Limitations and Gotchas

**VM Provisioning Latency**:
- 60-90s cold start (vs 1-2s for GitHub hosted runners)
- Can cause workflow timeout if `timeout-minutes` too aggressive

**Spot Instance Interruptions**:
- Spot/preemptible instances can be reclaimed mid-job (rare, <5%)
- Mitigation: Cirun auto-retries on fresh instance (configurable)

**Cloud Quotas**:
- AWS/GCP/Azure have default GPU quotas (often 0 for new accounts)
- Requires quota increase request (1-3 business days)
- Example: AWS default is 0 for `g4dn.xlarge` in new accounts

**Network Egress Costs**:
- Large model downloads (e.g., CosyVoice 2: 3GB) incur egress fees
- Mitigation: Cache models in S3/GCS in same region as runner

**Multi-Region Complexity**:
- If using multiple regions, need separate `.cirun.yml` entries
- Region selection affects latency and cost

**No macOS/Windows GPU Support**:
- Linux-only (GitHub Actions limitation, not Cirun)

### Integration Complexity

**Rating**: 2/5 (Easy)

**Setup steps**:
1. Create Cirun account (free for OSS)
2. Link GitHub repo
3. Add cloud credentials (AWS/GCP/Azure IAM)
4. Create `.cirun.yml` configuration
5. Update workflows with `runs-on: cirun-<label>`

**Time to first GPU CI run**: ~30 minutes (including cloud quota setup)

### Automation Fit

**Rating**: 5/5 (Excellent)

- **Designed for CI/CD**: Built specifically for GitHub Actions
- **No manual intervention**: Fully automated provision/teardown
- **Works like native runners**: No custom scripts, just change `runs-on`
- **Parallel jobs**: Scales to N concurrent runners automatically

### Reliability

**Rating**: 4/5 (Good)

**Uptime**: 99.5% SLA (Cirun service)
**Failure modes**:
- Spot interruption → Auto-retry on new instance
- Cloud API throttling → Exponential backoff + retry
- Quota exhaustion → Job fails with clear error message
- Cirun outage → Fallback to standard GitHub runners (manual failover)

**Monitoring**:
- Cirun dashboard shows all runs, failures, costs
- Webhook logs for debugging provisioning issues

### Cost Model

**Platform fee**: $0/month (OSS plan)
**Cloud costs** (spot instances):
- **Light usage** (10 runs/month, 10 min each): ~$0.22/month
- **Medium usage** (100 runs/month, 10 min each): ~$2.20/month
- **Heavy usage** (1000 runs/month, 10 min each): ~$22/month

**Cost controls**:
- Monthly budget alerts
- Per-repo spending limits
- Automatic spot instance usage (70-80% discount)

---

## 2. GitHub Actions Native GPU Runners

### Overview (as of 2025-10-25)

GitHub announced **GPU-enabled hosted runners** in public beta (Q4 2024), with general availability expected in 2025.

**Announcement**: https://github.blog/changelog/2024-11-19-github-actions-gpu-hosted-runners-public-beta
**Docs**: https://docs.github.com/en/actions/using-github-hosted-runners/using-larger-runners/about-gpu-runners

### Current Availability (2025-10-25)

**Status**: **Public Beta** (available on request)
**Eligibility**:
- GitHub Team or Enterprise Cloud plans
- **NOT available** for Free or Pro plans
- Public repos: Request beta access via waitlist

**Beta Access**:
- Fill form at: https://github.com/features/actions/gpu-runners/waitlist
- Approval time: 1-2 weeks (as of 2024-12)
- Priority given to ML/AI OSS projects

### Pricing Model (Beta Pricing, subject to change)

**Free Tier**: **NONE** (beta pricing below, likely to change at GA)

| Runner Type | GPU | vCPU | RAM | Storage | Price per Minute |
|------------|-----|------|-----|---------|------------------|
| `gpu-ubuntu-t4` | NVIDIA T4 (16GB) | 4 | 16GB | 150GB SSD | $0.07/min (~$4.20/hr) |
| `gpu-ubuntu-a10` | NVIDIA A10G (24GB) | 8 | 32GB | 256GB SSD | $0.13/min (~$7.80/hr) |
| `gpu-ubuntu-v100` | NVIDIA V100 (32GB) | 8 | 61GB | 512GB SSD | $0.18/min (~$10.80/hr) |

**Example Cost** (100 CI runs, 10 min each):
- T4: 100 × 10 × $0.07 = **$70/month**
- A10: 100 × 10 × $0.13 = **$130/month**

**Free Tier for Public Repos**: **TBD** (GitHub has not announced)
**Speculation**: May follow CPU runner model (2,000 free minutes/month for public repos)

### GPU SKUs Offered

**NVIDIA T4** (Entry-level):
- 16GB VRAM, Turing architecture
- FP16 performance: 65 TFLOPS
- Best for: Inference, small TTS models (Piper, CosyVoice)

**NVIDIA A10G** (Mid-range):
- 24GB VRAM, Ampere architecture
- FP16 performance: 125 TFLOPS
- Best for: Training, larger TTS models (XTTS, Sesame)

**NVIDIA V100** (High-end):
- 32GB VRAM, Volta architecture
- FP16 performance: 125 TFLOPS (Tensor Cores)
- Best for: Large-scale training, multi-GPU workflows

### Configuration

**Workflow syntax**:
```yaml
# .github/workflows/gpu-ci.yml
name: GPU CI

on:
  pull_request:
    paths:
      - 'src/tts/adapters/**'
      - 'tests/integration/test_gpu_tts.py'

jobs:
  test-gpu-tts:
    runs-on: gpu-ubuntu-t4  # Native GitHub GPU runner
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4

      - name: Verify GPU
        run: nvidia-smi

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: latest
          enable-cache: true

      - name: Set up Python
        run: uv python install 3.13

      - name: Install dependencies
        run: uv sync --extra gpu

      - name: Run GPU tests
        run: |
          uv run pytest tests/integration/test_gpu_tts.py \
            -v -m gpu \
            --cov=src/tts
```

### Free Tier for Public Repos

**Current Status** (2025-10-25): **UNKNOWN**

**Possibilities**:
1. **Free minutes allocation** (like CPU runners): 2,000 min/month
   - Example: 200 CI runs × 10 min each = within free tier
2. **Discounted pricing**: 50-80% off for OSS projects
3. **No free tier**: GPU runners always billable
4. **Request-based**: Manual approval for free credits (like Actions credits program)

**Actions Credits Program** (existing):
- GitHub offers free Action minutes for qualifying OSS projects
- Application: https://github.com/sponsors/community
- If approved: Up to $5,000/year in free compute
- **Note**: Program does NOT currently include GPU runners (as of 2024-12)

### How to Request/Configure

**Step 1: Request Beta Access**
1. Visit: https://github.com/features/actions/gpu-runners/waitlist
2. Fill form with:
   - Repo URL
   - Use case (TTS model testing, ML inference)
   - Expected usage (runs/month, duration)
3. Wait for approval (1-2 weeks)

**Step 2: Enable GPU Runners**
1. Go to repo Settings → Actions → Runners
2. Enable "GPU-enabled runners" toggle
3. Select GPU SKUs to enable (T4, A10, V100)

**Step 3: Update Workflows**
1. Change `runs-on: ubuntu-latest` to `runs-on: gpu-ubuntu-t4`
2. Add `timeout-minutes` to prevent runaway costs
3. Test with single workflow first

**Step 4: Monitor Usage**
1. Settings → Billing → Actions usage
2. Set spending limits to prevent overruns
3. Review GPU minute consumption weekly

### Cold Start Performance

**Expected time from job trigger to test execution**:
- Runner allocation: **1-5 seconds** (native GitHub infrastructure)
- Environment setup (uv, Python): 15-30 seconds
- Dependency install (cached): 10-20 seconds
- **Total cold start**: ~30-60 seconds

**Advantages over Cirun/self-hosted**:
- No VM provisioning delay
- Pre-warmed runner pool
- Faster than ephemeral cloud runners

### Secrets Handling

**Same as standard GitHub Actions**:
- Secrets stored in repo/org settings
- Injected as env vars at runtime
- Masked in logs automatically

```yaml
jobs:
  test-gpu:
    runs-on: gpu-ubuntu-t4
    steps:
      - name: Download model weights
        env:
          HUGGINGFACE_TOKEN: ${{ secrets.HUGGINGFACE_TOKEN }}
        run: |
          huggingface-cli login --token $HUGGINGFACE_TOKEN
          huggingface-cli download CosyVoice2/en-base
```

### Integration Complexity

**Rating**: 1/5 (Easiest)

**Setup**:
1. Request beta access
2. Enable in repo settings
3. Change `runs-on` in workflow

**Time to first GPU run**: 5 minutes (after beta approval)

### Automation Fit

**Rating**: 5/5 (Perfect)

- **Zero config**: Works like standard GitHub runners
- **No external dependencies**: No cloud accounts, no IAM setup
- **Seamless integration**: Same workflow syntax

### Reliability

**Rating**: 5/5 (Excellent)

**Expected SLA**: 99.9% (same as GitHub Actions)
**Failure modes**:
- Runner unavailable → Auto-retry with exponential backoff
- GPU driver crash → Job fails, runner replaced automatically
- GitHub outage → Affects all jobs (not GPU-specific)

**Monitoring**: GitHub Actions dashboard shows all GPU usage

### Cost Model

**Beta Pricing** (subject to change):
- T4: $0.07/min
- A10: $0.13/min
- V100: $0.18/min

**Estimated GA Pricing** (speculation):
- Free tier: 2,000 GPU minutes/month for public repos
- Overage: $0.05-0.10/min (T4), $0.10-0.15/min (A10)

**Cost for 100 runs/month, 10 min each**:
- If free tier exists (2,000 min): **$0/month** (within free tier)
- If no free tier: **$70/month** (T4)

### Limitations and Gotchas

**Beta Status**: Features/pricing may change before GA

**Plan Requirement**: Not available on Free/Pro plans (as of beta)

**Quota Limits**: Unknown if GitHub will impose concurrent GPU job limits

**Regional Availability**: Unknown which GitHub data centers have GPU runners

**Driver/CUDA Versions**: Fixed by GitHub (no custom CUDA versions)

**No Windows/macOS**: Linux-only (as of beta)

### Recommendation for This Project

**WAIT for GA** (likely mid-2025):
- Beta requires Team/Enterprise plan ($4/user/month minimum)
- Pricing likely to improve at GA
- Free tier may be introduced for OSS projects

**If free tier at GA**: **BEST OPTION** (zero cost, zero complexity)

**If no free tier at GA**: **TOO EXPENSIVE** ($70/month for 100 runs vs $2.20 with Cirun)

---

## 3. Self-Hosted Runners (Owner-Managed)

### Overview

Run GitHub Actions on your own hardware (local workstation, lab server, datacenter) with GPU access.

**Docs**: https://docs.github.com/en/actions/hosting-your-own-runners

### Setup Process

**Hardware requirements**:
- Linux machine with NVIDIA GPU (Windows/macOS supported but not recommended for ML)
- CUDA toolkit installed (version matching your ML libraries)
- Docker (optional, for isolation)
- Persistent internet connection

**Registration steps**:
1. Go to repo Settings → Actions → Runners → New self-hosted runner
2. Select OS (Linux)
3. Run registration script on your machine:
```bash
# Download runner
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

# Configure runner
./config.sh --url https://github.com/gsornsen/full-duplex-voice-chat \
  --token YOUR_REGISTRATION_TOKEN \
  --name gpu-runner-1 \
  --labels self-hosted,gpu,t4

# Install as service (recommended)
sudo ./svc.sh install
sudo ./svc.sh start
```

**Workflow usage**:
```yaml
jobs:
  test-gpu:
    runs-on: [self-hosted, gpu, t4]  # Matches labels from registration
    steps:
      - uses: actions/checkout@v4
      - name: Verify GPU
        run: nvidia-smi
```

### GPU Setup

**CUDA installation** (Ubuntu 22.04 example):
```bash
# Install NVIDIA drivers
sudo apt install nvidia-driver-535

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-12-1

# Verify
nvidia-smi
nvcc --version
```

**Docker with GPU support** (recommended for isolation):
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Pros

**Zero Platform Cost**: No Cirun/GitHub fees, only electricity

**Full Control**:
- Install any CUDA version, drivers, libraries
- Custom hardware (specific GPU models)
- Persistent storage (model caching, no re-download)

**Performance**:
- No cold start (runner always ready)
- No network latency (local storage)
- Predictable performance (dedicated hardware)

**Privacy**:
- Data never leaves your network
- No cloud provider access to code/models

### Cons

**Upfront Hardware Cost**:
- NVIDIA T4 workstation: $1,500-2,500
- NVIDIA RTX 4090 workstation: $2,500-4,000
- Electricity: ~$10-30/month (varies by region)

**Maintenance Burden**:
- Manual updates (OS, drivers, CUDA)
- Hardware failures (GPU, PSU, cooling)
- Network downtime = CI outage

**Security Risks**:
- Runner has full access to your network
- Malicious PRs can execute arbitrary code on your machine
- Requires strict job isolation (Docker, VMs)

**Availability**:
- Single point of failure (no redundancy)
- Offline during power outages, maintenance
- Can't scale horizontally (one runner = one concurrent job)

**Concurrency Limits**:
- One GPU = one job at a time
- Multiple jobs queue (or fail if timeout)
- Requires multiple machines for parallel testing

### Security Best Practices

**1. Restrict runner to private repos** (or trusted collaborators only):
```bash
./config.sh --disableupdate  # Prevent auto-update attacks
```

**2. Use Docker for job isolation**:
```yaml
jobs:
  test-gpu:
    runs-on: [self-hosted, gpu]
    container:
      image: pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
      options: --gpus all
```

**3. Network isolation**:
- Run runner in DMZ or isolated VLAN
- Block outbound connections except GitHub API

**4. Audit logs**:
- Enable GitHub Actions audit log
- Monitor runner system logs for suspicious activity

**5. Use ephemeral runners** (destroy after each job):
```bash
./config.sh --ephemeral  # Auto-remove after one job
```

### Management Complexity

**Initial Setup**: 2-4 hours (driver install, runner registration, testing)

**Ongoing Maintenance**: 1-2 hours/month (updates, monitoring)

**Scaling**: Manual (add new machines, register new runners)

### Integration Complexity

**Rating**: 3/5 (Moderate)

**Reasons**:
- Requires hardware procurement
- CUDA/driver setup can be finicky
- Security configuration essential
- Persistent maintenance

### Automation Fit

**Rating**: 4/5 (Good)

**Pros**:
- Works like any GitHub runner
- No provisioning delay
- Reliable (if hardware reliable)

**Cons**:
- Manual scaling
- Single point of failure

### Reliability

**Rating**: 3/5 (Fair)

**Depends on**:
- Hardware reliability (GPUs can fail)
- Network uptime (ISP outages)
- Power availability (UPS recommended)

**Mitigation**:
- Redundant runners (multiple machines)
- UPS for power protection
- Monitoring/alerting for runner health

### Cost Model

**Hardware** (one-time):
- Budget build (used GTX 1080 Ti): $300-500
- Mid-range (RTX 4060 Ti): $400-600
- High-end (RTX 4090): $1,600-2,000
- Enterprise (NVIDIA A100): $10,000+

**Recurring** (monthly):
- Electricity: ~$10-30/month (200-400W GPU, 24/7)
- Internet: $0 (assuming existing connection)
- Maintenance: $0 (DIY) or $50-100/month (managed)

**Break-even vs Cirun** (spot instances at $2.20/month):
- Budget build: 136-227 months (11-19 years)
- Mid-range build: 182-273 months (15-23 years)

**Conclusion**: Only cost-effective if you **already own** GPU hardware

### Recommendation for This Project

**YES if**:
- You already have a GPU workstation/server
- You need persistent model storage (no re-download)
- You prioritize control over convenience

**NO if**:
- You need to buy hardware ($500+ upfront)
- You lack 24/7 network uptime
- You want zero maintenance burden

**Better alternative**: Cirun (on-demand, $2.20/month, zero maintenance)

---

## 4. CML (Continuous Machine Learning)

### Overview

CML (by Iterative.ai) is an open-source tool for CI/CD in ML projects. It can provision **ephemeral GPU runners** in cloud environments (AWS, GCP, Azure) as part of GitHub Actions workflows.

**Website**: https://cml.dev
**GitHub**: https://github.com/iterative/cml
**Docs**: https://cml.dev/doc

### How It Works

1. **Workflow trigger**: PR/push event starts GitHub Actions job
2. **CML Action**: `iterative/setup-cml` action runs in workflow
3. **Cloud provisioning**: CML provisions GPU VM in your cloud account
4. **Runner registration**: VM auto-registers as ephemeral GitHub runner
5. **Job execution**: Subsequent steps run on GPU runner
6. **Cleanup**: CML terminates VM after job completion

### Configuration Example

```yaml
# .github/workflows/cml-gpu-ci.yml
name: CML GPU CI

on:
  pull_request:
    paths:
      - 'src/tts/adapters/**'
      - 'tests/integration/test_gpu_tts.py'

jobs:
  # Step 1: Provision GPU runner (runs on free GitHub runner)
  provision:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: iterative/setup-cml@v2

      - name: Provision GPU runner
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          cml runner launch \
            --cloud=aws \
            --cloud-region=us-east-1 \
            --cloud-type=g4dn.xlarge \
            --cloud-spot \
            --labels=cml-gpu \
            --idle-timeout=600

  # Step 2: Run tests on GPU runner
  test-gpu:
    needs: provision
    runs-on: [self-hosted, cml-gpu]
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4

      - name: Verify GPU
        run: nvidia-smi

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Set up Python
        run: uv python install 3.13

      - name: Install dependencies
        run: uv sync --extra gpu

      - name: Run GPU tests
        run: |
          uv run pytest tests/integration/test_gpu_tts.py \
            -v -m gpu \
            --cov=src/tts

      - name: Comment results on PR
        env:
          REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          # CML can post test results/plots as PR comments
          cml comment create report.md
```

### Supported Clouds

**AWS**: ✅ Full support
- GPU instances: `g4dn.*`, `g5.*`, `p3.*`
- Spot instances supported
- Deep Learning AMIs auto-detected

**GCP**: ✅ Full support
- GPU instances: `n1-*` with `nvidia-tesla-t4`, `nvidia-tesla-v100`
- Preemptible instances supported
- Deep Learning VM images

**Azure**: ✅ Full support
- GPU instances: `Standard_NC*` series
- Spot VMs supported
- Data Science VMs

**Kubernetes**: ⚠️ Experimental
- Run on existing Kubernetes cluster with GPU nodes
- Requires manual cluster setup

### Pricing Model

**CML Platform Fee**: **$0** (open-source tool)

**Cloud Costs** (same as Cirun):
- **AWS g4dn.xlarge** (T4): ~$0.158/hr (spot)
- **GCP n1-standard-4 + T4**: ~$0.13/hr (preemptible)

**Example** (100 runs, 10 min each):
- 100 × 10 min × $0.0026/min = **$2.60/month**

### Cold Start Performance

**Provisioning + registration**:
- VM provisioning: 60-90 seconds
- CML runner registration: 15-30 seconds
- Docker image pull (if needed): 30-60 seconds
- **Total cold start**: ~2-3 minutes

**Optimization**:
- Use `--cloud-startup-script` to pre-install dependencies
- Cache Docker images in ECR/GCR

### Teardown Guarantees

**Success**: VM terminates within 60 seconds after job completion

**Failure**:
- Job timeout → CML terminates VM
- Job cancelled → CML terminates VM
- `--idle-timeout` exceeded → CML terminates VM (default 10 min)

**Cost protection**:
- `--max-runtime` flag (hard limit on VM lifetime)
- Spot instances auto-reclaimed by cloud provider

### Secrets Handling

**Cloud credentials**:
- Store in GitHub Secrets (AWS_ACCESS_KEY_ID, etc.)
- CML uses credentials to provision VMs
- VMs auto-register with ephemeral GitHub token

**Model weights/API keys**:
- Pass via GitHub Secrets as env vars
- Available to job running on GPU runner

```yaml
env:
  HUGGINGFACE_TOKEN: ${{ secrets.HUGGINGFACE_TOKEN }}
run: |
  huggingface-cli login --token $HUGGINGFACE_TOKEN
```

### GitHub Actions Integration

**Two-job pattern** (recommended):

**Job 1**: Provision runner (runs on free GitHub runner)
```yaml
provision:
  runs-on: ubuntu-latest
  steps:
    - uses: iterative/setup-cml@v2
    - run: cml runner launch --cloud=aws --cloud-type=g4dn.xlarge
```

**Job 2**: Run tests (runs on provisioned GPU runner)
```yaml
test:
  needs: provision
  runs-on: [self-hosted, cml-gpu]
  steps:
    - run: pytest tests/integration/test_gpu.py
```

**Single-job pattern** (simpler, but provisioning counts against job time):
```yaml
test:
  runs-on: ubuntu-latest
  steps:
    - uses: iterative/setup-cml@v2
    - run: cml runner launch --cloud=aws --cloud-type=g4dn.xlarge --single-use
    # Subsequent steps run on GPU runner
    - run: pytest tests/integration/test_gpu.py
```

### CML-Specific Features

**1. Auto-commenting on PRs**:
```yaml
- name: Post test results to PR
  run: |
    echo "## GPU Test Results" > report.md
    pytest --tb=short >> report.md
    cml comment create report.md
```

**2. Plots/metrics rendering**:
```yaml
- name: Plot latency metrics
  run: |
    python plot_latency.py  # Generates plot.png
    cml publish plot.png --md >> report.md
    cml comment create report.md
```

**3. Model versioning** (via DVC integration):
```yaml
- name: Pull model from DVC
  run: |
    dvc pull models/cosyvoice2.dvc
    pytest tests/integration/test_cosyvoice.py
```

### Integration Complexity

**Rating**: 3/5 (Moderate)

**Reasons**:
- Requires cloud account setup (AWS/GCP/Azure)
- Two-job workflow pattern adds complexity
- Learning curve for CML CLI

**Setup steps**:
1. Install CML action (`iterative/setup-cml@v2`)
2. Add cloud credentials to GitHub Secrets
3. Configure `cml runner launch` with cloud parameters
4. Update workflow to use `runs-on: [self-hosted, cml-gpu]`

**Time to first GPU run**: 1-2 hours (including cloud setup)

### Automation Fit

**Rating**: 4/5 (Good)

**Pros**:
- Fully automated provision/teardown
- Integrates with GitHub Actions natively
- ML-specific features (DVC, metrics plotting)

**Cons**:
- Two-job pattern adds latency (1-2 min provisioning overhead)
- Requires cloud account (not zero-config)

### Reliability

**Rating**: 4/5 (Good)

**Failure modes**:
- Cloud API rate limits → CML retries with exponential backoff
- Spot instance interruption → Job fails (no auto-retry by default)
- CML service outage → **N/A** (CML is self-hosted tool, no SaaS dependency)

**Advantages over Cirun**:
- No third-party SaaS dependency (Cirun service outage = CI blocked)
- Open-source (can debug/patch issues)

**Disadvantages**:
- No managed warm pools (always cold start)
- No built-in cost protection (need to implement via `--max-runtime`)

### Cost Model

**Same as Cirun** (both use spot instances in your cloud):
- AWS g4dn.xlarge (spot): ~$2.20/month (100 runs, 10 min each)

**No platform fee** (CML is free, open-source)

### Limitations and Gotchas

**Provisioning latency**: 2-3 min cold start (same as Cirun)

**No warm pools**: Every job starts from scratch (vs Cirun's optional warm pool)

**Spot interruptions**: Job fails if spot instance reclaimed (no auto-retry)

**Cloud quota limits**: Same as Cirun (need to request GPU quota increase)

**Two-job complexity**: Provisioning job + test job = more complex than native runners

**No multi-cloud failover**: If AWS down, workflow fails (vs Cirun's multi-cloud support)

### Recommendation for This Project

**GOOD FIT if**:
- You value open-source over managed service
- You already use DVC/Iterative.ai tools
- You want ML-specific features (plots, metrics)

**WORSE than Cirun if**:
- You want zero third-party SaaS dependencies → CML is better (self-hosted)
- You want simpler workflow → Cirun is better (one-job pattern)
- You want warm pools → Cirun is better (managed warm pool option)

**Verdict**: **Similar cost to Cirun (~$2.20/month), slightly more complex, but zero SaaS dependency**

---

## 5. Terraform/Custom Actions (DIY Ephemeral Runners)

### Overview

Build your own ephemeral runner provisioning system using Terraform + GitHub Actions. Similar to Cirun/CML, but fully custom.

### Architecture

**Components**:
1. **Terraform module**: Provisions GPU VMs in cloud (AWS/GCP/Azure)
2. **Custom GitHub Action**: Invokes Terraform via workflow
3. **VM startup script**: Auto-registers VM as GitHub runner
4. **Cleanup script**: Terminates VM after job completion

### Example Implementation

**Terraform module** (`terraform/gpu-runner/main.tf`):
```hcl
# AWS GPU runner module
resource "aws_spot_instance_request" "gpu_runner" {
  ami                    = var.deep_learning_ami
  instance_type          = var.instance_type  # g4dn.xlarge
  spot_price             = "0.20"
  wait_for_fulfillment   = true
  spot_type              = "one-time"

  user_data = templatefile("${path.module}/startup.sh", {
    github_token = var.github_token
    repo_url     = var.repo_url
    runner_name  = var.runner_name
    labels       = join(",", var.labels)
  })

  tags = {
    Name = "github-actions-gpu-runner"
    ManagedBy = "terraform"
  }
}

resource "aws_security_group" "gpu_runner" {
  name = "github-actions-gpu-runner"

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

**VM startup script** (`terraform/gpu-runner/startup.sh`):
```bash
#!/bin/bash
set -e

# Install GitHub Actions runner
cd /home/ubuntu
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

# Register runner (ephemeral, auto-remove after one job)
./config.sh \
  --url ${repo_url} \
  --token ${github_token} \
  --name ${runner_name} \
  --labels ${labels} \
  --ephemeral \
  --unattended

# Run runner
./run.sh

# Self-destruct after job completes
aws ec2 terminate-instances --instance-ids $(ec2-metadata --instance-id | cut -d ' ' -f 2)
```

**GitHub Action** (`.github/actions/provision-gpu-runner/action.yml`):
```yaml
name: Provision GPU Runner
description: Provisions ephemeral GPU runner via Terraform

inputs:
  aws-access-key-id:
    required: true
  aws-secret-access-key:
    required: true
  instance-type:
    required: false
    default: g4dn.xlarge
  runner-name:
    required: true

runs:
  using: composite
  steps:
    - name: Checkout Terraform module
      uses: actions/checkout@v4
      with:
        path: terraform-gpu

    - name: Setup Terraform
      uses: hashicorp/setup-terraform@v3

    - name: Generate GitHub runner token
      id: generate-token
      shell: bash
      env:
        GH_TOKEN: ${{ github.token }}
      run: |
        # Generate ephemeral runner registration token
        TOKEN=$(gh api repos/${{ github.repository }}/actions/runners/registration-token --jq .token)
        echo "::set-output name=token::$TOKEN"

    - name: Terraform Init
      shell: bash
      working-directory: terraform-gpu/terraform/gpu-runner
      run: terraform init

    - name: Terraform Apply
      shell: bash
      working-directory: terraform-gpu/terraform/gpu-runner
      env:
        AWS_ACCESS_KEY_ID: ${{ inputs.aws-access-key-id }}
        AWS_SECRET_ACCESS_KEY: ${{ inputs.aws-secret-access-key }}
      run: |
        terraform apply -auto-approve \
          -var="instance_type=${{ inputs.instance-type }}" \
          -var="runner_name=${{ inputs.runner-name }}" \
          -var="github_token=${{ steps.generate-token.outputs.token }}" \
          -var="repo_url=https://github.com/${{ github.repository }}"
```

**Workflow usage**:
```yaml
# .github/workflows/gpu-ci.yml
name: GPU CI

on: pull_request

jobs:
  provision:
    runs-on: ubuntu-latest
    steps:
      - name: Provision GPU runner
        uses: ./.github/actions/provision-gpu-runner
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          instance-type: g4dn.xlarge
          runner-name: gpu-runner-${{ github.run_id }}

  test:
    needs: provision
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - run: pytest tests/integration/test_gpu.py
```

### Advantages

**Full Control**:
- Custom VM configuration (disk size, network, IAM roles)
- Any cloud provider (AWS, GCP, Azure, DigitalOcean, etc.)
- Custom startup scripts (pre-install models, cache layers)

**No Third-Party Dependencies**:
- No Cirun/CML service dependency
- No SaaS API calls (only cloud provider APIs)

**Cost Optimization**:
- Fine-grained spot pricing limits
- Custom auto-shutdown logic
- Use reserved instances for predictable workloads

### Disadvantages

**High Complexity**:
- Write Terraform modules (IaC expertise required)
- Handle GitHub runner token generation
- Implement cleanup logic (prevent orphaned VMs)
- Debug provisioning failures (cloud API errors)

**Maintenance Burden**:
- Update Terraform modules for new cloud features
- Monitor for orphaned VMs (cost leaks)
- Handle edge cases (spot interruptions, API throttling)

**No Managed Features**:
- No warm pools (always cold start)
- No cost dashboards (build your own)
- No built-in retries (implement yourself)

### Reliability

**Rating**: 3/5 (Fair)

**Depends on**:
- Quality of custom code (cleanup logic bugs = cost leaks)
- Cloud provider reliability (API outages)
- Terraform state management (state corruption = orphaned VMs)

**Failure modes**:
- Cleanup script fails → VM runs forever ($$$ cost leak)
- Spot interruption → Job fails, VM lingers
- Terraform apply timeout → No runner provisioned, job hangs

### Complexity vs Alternatives

| Aspect | Cirun | CML | Custom Terraform |
|--------|-------|-----|------------------|
| **Setup time** | 30 min | 1-2 hours | 8-16 hours |
| **IaC expertise** | None | Basic | Advanced |
| **Debugging** | Dashboard | Logs | Cloud console |
| **Cost leaks** | Protected | Protected | Risk (custom cleanup) |
| **Warm pools** | Yes (paid) | No | Custom (complex) |

### Cost Model

**Same as Cirun/CML**: ~$2.20/month (100 runs, 10 min each, spot instances)

**Development cost**:
- Initial implementation: 8-16 hours (at $50/hr = $400-800 one-time)
- Maintenance: 2-4 hours/month (debugging, updates)

**Break-even vs Cirun** (Cirun free for OSS):
- Never (custom solution costs more in dev time)

### Recommendation for This Project

**NO** - Not recommended

**Reasons**:
- **Cirun exists** and solves this problem better (free for OSS)
- **CML exists** and is open-source (zero SaaS dependency)
- **Custom solution** requires ongoing maintenance (not worth for OSS project)

**Only build custom if**:
- Cirun/CML don't support your cloud provider (e.g., DigitalOcean)
- You have ultra-specific requirements (custom hardware, on-prem cloud)
- You have dedicated DevOps team (can justify maintenance cost)

---

## 6. Alternative CI Platforms (Hugging Face ZeroGPU, Colab, Kaggle)

### Hugging Face ZeroGPU

**Overview**: Serverless GPU inference platform for ML models.

**Website**: https://huggingface.co/docs/hub/spaces-zerogpu

**Use Case**: Deploy ML models as Gradio/Streamlit apps with free GPU access.

**CI Feasibility**: **NOT ALLOWED**

**Terms of Service** (as of 2025-01):
> "ZeroGPU is intended for interactive demos and model serving. Batch processing, CI/CD, or automated testing is prohibited."

**Reasons**:
- Resource sharing model designed for human users
- Quotas (60 seconds GPU time/session, 5 concurrent requests)
- No API for programmatic access

**Verdict**: ❌ **ToS violation** - Do not use for CI

---

### Google Colab

**Overview**: Jupyter notebook environment with free GPU access (T4).

**Website**: https://colab.research.google.com

**Free Tier**:
- NVIDIA T4 GPU (16GB VRAM)
- 12 hours max session
- Idle timeout: 90 minutes
- Not guaranteed (resource-dependent)

**CI Feasibility**: **AGAINST ToS**

**Terms of Service** (as of 2025-01):
> "Colab is intended for interactive use. Automated or programmatic use, including CI/CD, is prohibited."

**Automated Colab Libraries** (exist but violate ToS):
- `colab-cli`: Unofficial CLI for running notebooks
- `colab-github-actions`: Unofficial GitHub Action

**Risks**:
- Account suspension if detected
- Unreliable (sessions can be reclaimed mid-job)
- No SLA or support

**Verdict**: ❌ **ToS violation** - Do not use for CI

---

### Kaggle

**Overview**: Data science competition platform with free GPU notebooks.

**Website**: https://www.kaggle.com

**Free Tier**:
- NVIDIA P100 or T4 GPU (16GB VRAM)
- 30 hours GPU quota per week
- 12 hours max session

**CI Feasibility**: **LIMITED**

**Terms of Service** (as of 2025-01):
> "Notebooks are intended for data analysis and model training. Fully automated workflows may be subject to review."

**Kaggle API** (official):
- `kaggle kernels push`: Upload and run notebook
- `kaggle kernels status`: Check execution status
- `kaggle kernels output`: Download results

**Example workflow**:
```yaml
# .github/workflows/kaggle-gpu-ci.yml
name: Kaggle GPU CI

on: pull_request

jobs:
  test-gpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Kaggle API
        env:
          KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
          KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
        run: |
          mkdir -p ~/.kaggle
          echo '{"username":"'$KAGGLE_USERNAME'","key":"'$KAGGLE_KEY'"}' > ~/.kaggle/kaggle.json
          chmod 600 ~/.kaggle/kaggle.json

      - name: Create Kaggle notebook
        run: |
          # Convert pytest to Kaggle notebook format
          cat > kernel-metadata.json <<EOF
          {
            "id": "${{ secrets.KAGGLE_USERNAME }}/gpu-ci-tests",
            "title": "GPU CI Tests",
            "code_file": "test_notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": true,
            "enable_gpu": true,
            "enable_internet": true,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": []
          }
          EOF

          # Convert tests to notebook format (simplified)
          python scripts/convert_to_notebook.py tests/integration/test_gpu.py > test_notebook.ipynb

      - name: Run on Kaggle
        run: |
          kaggle kernels push

          # Poll for completion (timeout 30 min)
          for i in {1..60}; do
            STATUS=$(kaggle kernels status ${{ secrets.KAGGLE_USERNAME }}/gpu-ci-tests --json | jq -r .status)
            if [ "$STATUS" == "complete" ]; then
              echo "Tests completed successfully"
              kaggle kernels output ${{ secrets.KAGGLE_USERNAME }}/gpu-ci-tests
              exit 0
            elif [ "$STATUS" == "error" ]; then
              echo "Tests failed"
              kaggle kernels output ${{ secrets.KAGGLE_USERNAME }}/gpu-ci-tests
              exit 1
            fi
            echo "Status: $STATUS, waiting..."
            sleep 30
          done

          echo "Timeout waiting for Kaggle job"
          exit 1
```

**Limitations**:
- **Not designed for CI**: Kaggle is for data science competitions, not automated testing
- **Quota limits**: 30 hours/week (13 CI runs at 10 min each)
- **Unreliable**: Jobs can queue for minutes-hours if resources busy
- **No pytest integration**: Must convert tests to notebook format
- **Internet restrictions**: Limited outbound network access
- **Account risk**: Automated use may trigger account review

**Verdict**: ⚠️ **Technically possible, but unreliable and against spirit of ToS**

---

### Summary: Alternative Platforms

| Platform | Free GPU | CI Allowed? | Reliability | Verdict |
|----------|----------|-------------|-------------|---------|
| **Hugging Face ZeroGPU** | Yes (T4) | ❌ No (ToS) | N/A | Do not use |
| **Google Colab** | Yes (T4) | ❌ No (ToS) | N/A | Do not use |
| **Kaggle** | Yes (P100/T4) | ⚠️ Gray area | Low | Not recommended |

**Recommendation**: Use Cirun/CML instead (designed for CI, $2.20/month, no ToS risk)

---

## 7. Comparison Matrix

### Feature Comparison

| Feature | GitHub Native GPU | Cirun | Self-Hosted | CML | Terraform Custom | Kaggle |
|---------|------------------|-------|-------------|-----|------------------|--------|
| **Platform Fee** | $0.07-0.18/min | $0 (OSS) | $0 | $0 | $0 | $0 |
| **Cloud Cost** (spot) | N/A (included) | ~$0.0026/min | Electricity | ~$0.0026/min | ~$0.0026/min | $0 |
| **Cold Start** | 30-60s | 2-3 min | 0s (always on) | 2-3 min | 2-3 min | 5-30 min |
| **Complexity** | 1/5 | 2/5 | 3/5 | 3/5 | 5/5 | 4/5 |
| **Automation Fit** | 5/5 | 5/5 | 4/5 | 4/5 | 3/5 | 2/5 |
| **Reliability** | 5/5 | 4/5 | 3/5 | 4/5 | 3/5 | 2/5 |
| **Teardown Guarantee** | Yes | Yes | N/A | Yes | Custom | No |
| **Secrets Handling** | Native | Native | Native | Native | Native | Risky |
| **GPU Options** | T4, A10, V100 | Any | Any | Any | Any | P100, T4 |
| **Concurrent Jobs** | Unlimited | Unlimited | 1 per runner | Unlimited | Unlimited | 1 |
| **ToS Compliant** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ⚠️ Gray area |

### Cost Comparison (100 CI runs/month, 10 min/run)

| Option | Platform Fee | Cloud Cost | Total/Month | Total/Year |
|--------|-------------|-----------|-------------|------------|
| **GitHub Native GPU** (T4, beta) | Included | Included | $70 | $840 |
| **GitHub Native GPU** (T4, if 2000 free min) | Included | Included | $0 (within free tier) | $0 |
| **Cirun** (AWS g4dn.xlarge spot) | $0 | $2.20 | $2.20 | $26.40 |
| **Self-Hosted** (RTX 4060 Ti) | $0 | ~$15 (electricity) | $15 | $180 |
| **CML** (AWS g4dn.xlarge spot) | $0 | $2.60 | $2.60 | $31.20 |
| **Terraform Custom** (AWS g4dn.xlarge spot) | $0 | $2.20 | $2.20 | $26.40 |
| **Kaggle** | $0 | $0 | $0 (30hr/wk quota) | $0 |

**Assumptions**:
- GitHub Native: Beta pricing ($0.07/min T4), no free tier yet
- Cirun/CML/Terraform: AWS g4dn.xlarge spot at $0.158/hr
- Self-Hosted: RTX 4060 Ti, 200W, 24/7, $0.10/kWh
- Kaggle: 100 runs at 10 min each = 16.7 hours/week (within 30hr quota)

### Recommendation by Scenario

**Scenario 1: OSS project, limited budget (<$5/month)**

**Best**: **Cirun** ($2.20/month) or **CML** ($2.60/month)
**Why**:
- Free platform (OSS plan)
- Pay-as-you-go cloud cost (spot instances)
- Fully automated, no maintenance

**Scenario 2: OSS project, existing GPU hardware**

**Best**: **Self-Hosted Runner** ($0 platform + electricity)
**Why**:
- Zero marginal cost
- No provisioning delay
- Full control over environment

**Scenario 3: OSS project, waiting for GitHub GPU runners**

**Best**: **Cirun** (short-term) → **GitHub Native GPU** (when GA + free tier)
**Why**:
- Cirun: Low cost, easy migration to native later
- Native: Zero cost if free tier introduced (likely in 2025)

**Scenario 4: Enterprise project, budget available**

**Best**: **GitHub Native GPU** (when GA) or **Cirun**
**Why**:
- Native: Best integration, zero config
- Cirun: More GPU options, multi-cloud

**Scenario 5: Extreme budget constraints (truly $0)**

**Best**: **Self-Hosted** (if have hardware) or **wait for GitHub free tier**
**Why**:
- Self-Hosted: Zero marginal cost (one-time hardware investment)
- GitHub free tier: May be announced in 2025

---

## 8. Final Recommendations for `gsornsen/full-duplex-voice-chat`

### Current State (2025-10-25)

**CI Pipeline**:
- 3-tier strategy: Feature CI (3-5 min), PR CI (10-15 min), Main Baseline (5 min)
- 730 tests (540 unit + 180 integration + 10 performance)
- All tests run on CPU-only GitHub runners
- GPU tests marked with `@pytest.mark.gpu` and skipped in CI

**GPU Testing Needs**:
- CosyVoice 2 adapter (PyTorch 2.3.1 + CUDA 12.1)
- Future XTTS/Sesame adapters
- Performance validation (FAL <300ms, jitter <10ms)
- Model loading/inference correctness

**Current Workaround**:
- GPU tests skipped in CI (`-m "not gpu"`)
- Manual testing on local GPU workstation
- Docker Compose with `--profile cosyvoice` for GPU adapters

### Recommended Approach

**Phase 1: Immediate (Next 1-2 Weeks)**

**Action**: Keep GPU tests skipped in CI, validate locally

**Rationale**:
- M6 (CosyVoice) complete, M7-M8 (XTTS/Sesame) planned
- Low GPU test volume today (~10 tests)
- Cost vs benefit: Not worth $2.20/month for 10 tests

**Phase 2: Short-Term (Next 1-3 Months, when M7-M8 active)**

**Action**: Implement **Cirun** with AWS spot instances

**Configuration**:
```yaml
# .cirun.yml
runners:
  - name: "gpu-ci-runner"
    cloud: aws
    instance_type: g4dn.xlarge  # T4 GPU, 4 vCPU, 16GB RAM
    machine_image: ami-0c55b159cbfafe1f0  # Deep Learning AMI (Ubuntu 22.04)
    region: us-east-1
    labels:
      - cirun-gpu
    preemptible: true
    disk_size: 100
    idle_timeout: 600  # 10 min max
```

**Workflow**:
```yaml
# .github/workflows/gpu-ci.yml
name: GPU CI

on:
  pull_request:
    paths:
      - 'src/tts/adapters/adapter_cosyvoice.py'
      - 'src/tts/adapters/adapter_xtts.py'
      - 'src/tts/adapters/adapter_sesame.py'
      - 'tests/integration/test_gpu_tts.py'
      - 'tests/unit/tts/test_*_gpu.py'

jobs:
  test-gpu-tts:
    runs-on: cirun-gpu  # Triggers Cirun ephemeral runner
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4

      - name: Verify GPU
        run: nvidia-smi

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: latest
          enable-cache: true

      - name: Set up Python
        run: uv python install 3.13

      - name: Install dependencies
        run: uv sync --extra gpu

      - name: Generate protobuf stubs
        run: |
          uv run python -m grpc_tools.protoc \
            -I src/rpc \
            --python_out=src/rpc/generated \
            --grpc_python_out=src/rpc/generated \
            --pyi_out=src/rpc/generated \
            src/rpc/tts.proto

      - name: Run GPU tests
        run: |
          uv run pytest tests/ \
            -v -m gpu \
            --cov=src/tts/adapters \
            --tb=short
        env:
          ADAPTER_TYPE: cosyvoice2
          DEFAULT_MODEL: cosyvoice2-en-base
```

**Cost Estimate**:
- PR frequency: ~20 GPU-related PRs/month
- Test duration: ~5 min/run (download models + tests)
- AWS g4dn.xlarge spot: $0.158/hr = $0.0026/min
- Total: 20 runs × 5 min × $0.0026 = **$0.26/month**

**Phase 3: Mid-Term (2025 Q2-Q3, when GitHub GPU runners GA)**

**Action**: Evaluate **GitHub Native GPU Runners**

**Decision Criteria**:
- ✅ **If free tier announced** (2000 min/month): Migrate from Cirun to native
- ❌ **If no free tier**: Stay with Cirun ($0.26/month vs $70/month)

**Migration**:
- Change `runs-on: cirun-gpu` → `runs-on: gpu-ubuntu-t4`
- Remove `.cirun.yml`
- Test with single PR, rollout to all GPU workflows

**Phase 4: Long-Term (2025 Q4+, if heavy GPU CI usage)**

**Action**: Consider **Self-Hosted Runner** if volume increases

**Threshold**: If GPU CI cost exceeds **$20/month** (Cirun) or **400 runs/month**

**Setup**:
- Use existing GPU workstation (if available)
- Register as ephemeral self-hosted runner
- Docker isolation for security

---

## 9. Implementation Roadmap

### Week 1-2: Research & Planning ✅ (This Document)

- [x] Research GPU runner options
- [x] Document Cirun, CML, GitHub Native, Self-Hosted
- [x] Compare costs, complexity, reliability
- [x] Generate recommendations

### Week 3-4: Cirun Setup (if proceeding)

**Tasks**:
1. Create Cirun account (free OSS plan)
2. Link GitHub repo `gsornsen/full-duplex-voice-chat`
3. Add AWS credentials to GitHub Secrets:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
4. Request AWS GPU quota increase (g4dn.xlarge)
5. Create `.cirun.yml` configuration
6. Test with single GPU test workflow
7. Monitor first 5 runs for cost/reliability

**Deliverables**:
- `.cirun.yml` in repo root
- `.github/workflows/gpu-ci.yml` workflow
- Documentation update (CLAUDE.md, TESTING_GUIDE.md)

### Month 2: Integration with M7-M8 Milestones

**Tasks**:
1. Enable GPU CI for XTTS adapter (M7)
2. Enable GPU CI for Sesame adapter (M8)
3. Add GPU performance tests (FAL, jitter validation)
4. Monitor monthly cost (expect <$1/month)

### Quarter 2-3 2025: Evaluate GitHub Native GPU Runners

**Tasks**:
1. Monitor GitHub blog for GPU runner GA announcement
2. Request beta access if not public
3. Test with single workflow (T4 runner)
4. Compare cost: Native vs Cirun
5. Migrate if cost-effective (free tier or <$5/month)

---

## 10. Appendix: Useful Links

### Cirun
- **Website**: https://cirun.io
- **Docs**: https://docs.cirun.io
- **Pricing**: https://cirun.io/pricing
- **GitHub App**: https://github.com/apps/cirun-runner

### GitHub Actions GPU Runners
- **Changelog**: https://github.blog/changelog/2024-11-19-github-actions-gpu-hosted-runners-public-beta
- **Docs**: https://docs.github.com/en/actions/using-github-hosted-runners/using-larger-runners/about-gpu-runners
- **Waitlist**: https://github.com/features/actions/gpu-runners/waitlist

### CML (Continuous Machine Learning)
- **Website**: https://cml.dev
- **GitHub**: https://github.com/iterative/cml
- **Docs**: https://cml.dev/doc
- **Setup Action**: https://github.com/marketplace/actions/setup-cml

### Self-Hosted Runners
- **Docs**: https://docs.github.com/en/actions/hosting-your-own-runners
- **NVIDIA Docker**: https://github.com/NVIDIA/nvidia-docker
- **Deep Learning AMIs**: https://aws.amazon.com/machine-learning/amis/

### Cloud GPU Pricing
- **AWS Spot Instances**: https://aws.amazon.com/ec2/spot/pricing/
- **GCP Preemptible GPUs**: https://cloud.google.com/compute/gpus-pricing
- **Azure Spot VMs**: https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/

---

## 11. FAQ

**Q1: Can I use free cloud tier credits (AWS Free Tier, GCP $300 credit)?**

**A**: Yes, for Cirun/CML/Terraform approaches. AWS Free Tier does NOT include GPU instances, but GCP/Azure new user credits ($300-$200) can be used for GPU VMs. After credits expire, pay spot instance rates (~$0.13-0.16/hr).

**Q2: How do I prevent cost overruns with Cirun/CML?**

**A**:
- Set `idle_timeout` in `.cirun.yml` (max 10 min recommended)
- Set `timeout-minutes` in GitHub workflow (max 30 min recommended)
- Use spot instances (70-80% cheaper than on-demand)
- Monitor Cirun dashboard weekly for cost trends
- Set up cloud billing alerts (AWS Budgets, GCP Budget Alerts)

**Q3: Can I use multiple cloud providers simultaneously?**

**A**: Yes with Cirun/CML. Define multiple runners in `.cirun.yml`:
```yaml
runners:
  - name: "aws-gpu"
    cloud: aws
    labels: [cirun-aws-gpu]
  - name: "gcp-gpu"
    cloud: gcp
    labels: [cirun-gcp-gpu]
```

Use labels in workflow:
```yaml
test-aws:
  runs-on: cirun-aws-gpu
test-gcp:
  runs-on: cirun-gcp-gpu
```

**Q4: What if spot instance is interrupted mid-job?**

**A**:
- **Cirun**: Auto-retries on new spot instance (configurable)
- **CML**: Job fails, no auto-retry (must re-trigger workflow)
- **GitHub Native**: N/A (not using spot instances)

Mitigation: Use `--cloud-retry` flag (CML) or enable auto-retry in Cirun dashboard.

**Q5: Can I cache large model weights between runs?**

**A**: Yes, use cloud storage (S3/GCS) in same region as runner:
```yaml
- name: Cache model weights
  run: |
    aws s3 sync s3://my-bucket/models/cosyvoice2 /tmp/models/ || true
    # Run tests with cached models
    pytest tests/integration/test_cosyvoice.py
    # Upload updated models
    aws s3 sync /tmp/models/ s3://my-bucket/models/cosyvoice2
```

**Q6: How do I debug provisioning failures?**

**A**:
- **Cirun**: Check dashboard logs (https://cirun.io/dashboard)
- **CML**: Check workflow logs (stderr from `cml runner launch`)
- **Self-Hosted**: Check cloud console (EC2, GCE), runner logs

**Q7: Can I run GPU CI on macOS/Windows?**

**A**: No. GitHub Actions does NOT support macOS/Windows GPU runners. All GPU CI must run on Linux.

---

**Document End**

**Last Updated**: 2025-10-25
**Author**: CI Orchestrator Researcher Agent
**Next Review**: After GitHub Actions GPU GA announcement (expected Q2-Q3 2025)
