# mDeBERTa-v3 Based Soft Prompt Tuning for CMCS Text Classification

This project implements a highly parameter-efficient approach for classifying Code-Mixed and Code-Switched (CMCS) text (specifically Kannada-English) using Soft Prompt Tuning with **mDeBERTa-v3** as the backbone. It is an extension of the methodologies presented in the research paper *"Use of prompt-based learning for code-mixed and code-switched text classification" (Udawatta et al., 2024)*.

## 🌟 Key Features

* **Extreme Parameter Efficiency:** Freezes the entire 278M-parameter mDeBERTa-v3 backbone and trains only ~25,600 soft prompt/verbalizer parameters — a **99.99% reduction** in trainable parameters while matching or exceeding the performance of adapter-based methods.
* **Disentangled Attention:** Leverages mDeBERTa's advanced attention mechanism, which separates content and positional representations. This natively handles the unpredictable script and positional patterns found in code-mixed text.
* **Dynamic Dataset Handling:** Automatically parses CSV/Excel datasets, dynamically discovering the target text columns and unique classification labels without requiring hardcoded configurations.
* **Ironclad Compatibility Shield:** A robust set of runtime patches that ensures the OpenPrompt framework remains fully functional on Google Colab despite evolving environments (NumPy 2.0 deprecations, Transformers cache changes, etc.).

## 📊 Results Summary

Performance evaluated on the Kannada-English CMCS Dataset:

| Task | mDeBERTa SP+SV (Ours) | XLM-R SP+SV [48] | Parameters Trained |
| :--- | :--- | :--- | :--- |
| **Hate-Speech Detection** | **78.0** F1 | 74.0 F1 | ~25.6K |
| **Sentiment Classification** | **66.4** F1 | 62.5 F1 | ~25.6K |

## 🚀 How to Run (Google Colab)

This repository is optimized for execution on Google Colab.

1. Create a new Google Colab Notebook.
2. Upload the `colab_mdeberta_extension.py` script and your dataset (`.csv` or `.xlsx`) to the Colab environment.
3. Add the following to a cell and run it:

```python
import colab_mdeberta_extension

# Run the automated setup and training pipeline
pipeline = colab_mdeberta_extension.AutomatedPipeline(
    dataset_path="your_dataset.csv", # Replace with your dataset filename
    max_epochs=10
)
pipeline.run_all()
```

The script will automatically:
* Install the required pip packages (`openprompt`, `transformers==4.38.2`, etc.).
* Restart the runtime if necessary (due to PyTorch/Transformers updates).
* Patch all environment incompatibilities automatically via the Ironclad Shield.
* Dynamically discover your dataset's labels, map them, and start training the Soft Prompts.

## 📂 Project Structure

* `colab_mdeberta_extension.py`: The main entry point. Contains the Ironclad Compatibility Shield and the automated PyTorch training loop designed specifically for cloud environments.
* `mdeberta_prompt_tuning.py`: Defines the `SoftTemplate` and `SoftVerbalizer` logic, registering mDeBERTa-v3 properly within the OpenPrompt architecture.
* `cmcs_dataset_handler.py`: Contains the `Dynamic Label Discovery` algorithm for reading arbitrary classification datasets and converting them into OpenPrompt's `InputExample` format.
* `demo_mdeberta.py`: An interactive demo module to simulate text predictions and display comparative result tables.

## 👥 Authors
* **K. Harsha Vardhan**
* **K. Shyam Sundar**
* **K. Sai Srikanth**

**Project Guide:** Dr. M. Sreelatha  
**Institution:** Department of Computer Science and Engineering, R.V.R & J.C. College of Engineering, Chowdavaram, Guntur, Andhra Pradesh, India.
