# CropShield AI — Presentation-Ready AI + IoT Agriculture Platform

## What is included
- Premium responsive web dashboard
- Farmer-friendly multilingual UI: English, Telugu, Hindi, Tamil, Kannada, Malayalam
- Browser microphone/voice navigation and voice output
- Crop image analysis endpoint
- Thermal stress analysis module
- IoT sensor dashboard and water-need prediction
- Automated irrigation gate controls
- Trained moisture prediction ensemble (`models/moisture_ensemble.joblib`)
- Multi-disease PlantVillage training pipeline covering all 38 classes
- Dataset manifest for additional agricultural datasets

## Why you may see different confidence values
Confidence is NOT fixed at 72%. When a trained disease model is present, the value comes from the model. When it is not present, the app explicitly labels the result as **Prototype image-evidence inference** instead of presenting it as a trained disease classifier.

## First run on Windows
1. Extract the ZIP.
2. Open the folder in VS Code.
3. Run `run_first_time.bat` or use the terminal steps in this README.
4. The setup script installs dependencies, downloads PlantVillage through Hugging Face Datasets, trains a 38-class ensemble using 200 images/class by default (up to 7,600 images), then starts the application.

For the largest PlantVillage training run, use `run_max_training.bat`. This uses the full available class data subject to the dataset contents and your machine's RAM/storage/time.

## Dataset strategy
PlantVillage has 54,303 images in 38 classes and is the primary multi-crop dataset. Paddy Doctor contributes 16,225 annotated paddy leaf images across 13 classes. A separate rice multi-source public dataset advertises 30,000+ images across 17 disease classes plus healthy/pest. These are tracked in `data/disease/dataset_manifest.json` rather than copied into the ZIP, because the image archives are large and dataset licenses/redistribution terms vary.

For a defensible research result, keep external datasets for validation and report macro-F1, per-class recall, and a confusion matrix. Do not claim those external datasets were used for training unless you actually run the corresponding download/processing scripts.

## Current model boundary
The included moisture model is trained and packaged. The disease model training is reproducible but the large image archive is not embedded. If you run the training step once on your PC, `models/disease_ensemble.joblib` will be created and the website will automatically use it.
