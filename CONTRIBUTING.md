# Contributing to AdverSynth

Thank you for your interest in contributing to **AdverSynth**! We welcome contributions from clinical researchers, machine learning practitioners, and open-source software engineers.

---

## 🛠️ Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/Ravva-78/adversynth.git
   cd adversynth
   ```

2. **Environment Configuration**
   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install pytest flake8 pytest-cov
   ```

4. **Verify Installation**
   Run the test suite to ensure your environment is functioning properly:
   ```bash
   pytest -v
   ```

---

## 📐 Code Style & Quality Standards

- **Formatting & Linting**: We adhere to PEP 8 standards. Run `flake8` before submitting changes:
  ```bash
  flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
  ```
- **Modular Design**: Keep network architectures in `models/`, training loops in `training/`, and metric calculations in `evaluation/`.
- **Reproducibility**: Ensure all random seed parameters (`torch.manual_seed`, `np.random.seed`) are controllable via config files or CLI arguments.
- **Unit Testing**: Any new model, loss function, or data transform should include corresponding test coverage under `tests/`.

---

## 🚀 Pull Request Workflow

1. Create a feature branch with a descriptive name:
   ```bash
   git checkout -b feat/add-stylegan-generator
   ```
2. Commit your modifications with clear, semantic commit messages:
   ```bash
   git commit -m "feat(models): add progressive growing generator block"
   ```
3. Run the complete test suite locally:
   ```bash
   pytest -v
   ```
4. Push to your fork and submit a Pull Request targeting the `main` branch.
5. Ensure all CI automated checks pass on GitHub Actions.

---

## 📋 Reporting Issues

- For bug reports or feature requests, please use the predefined [GitHub Issue Templates](.github/ISSUE_TEMPLATE/).
- Provide minimal reproducible examples with model hyperparameters, PyTorch versions, and hardware device specs (CUDA/CPU).
