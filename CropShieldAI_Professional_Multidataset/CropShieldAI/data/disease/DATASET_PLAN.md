# Crop Disease Dataset

## Primary dataset: PlantVillage
PlantVillage contains 54,304 labeled leaf images across 38 plant/disease or healthy classes. It is a strong baseline dataset for multi-disease classification, but images are largely captured under controlled conditions.

For this project, `training/train_disease_ensemble.py` automatically downloads the PlantVillage `color` dataset through Hugging Face and trains a lightweight image-feature ensemble.

Default training setting is presentation-friendly: up to 80 images per class (3,040 images maximum) with a stratified 80/20 split. Increase `--per-class` on a stronger PC/GPU for a larger model.

## Secondary validation dataset: PlantDoc
PlantDoc has 2,598 field-style images across 13 plant species and up to 17 disease classes. It is useful for testing generalization beyond controlled-background imagery. It is not bundled in this ZIP because of size/source distribution constraints.

## Rice-specific extension: Paddy Doctor
Paddy Doctor contains 16,225 annotated paddy leaf images across 13 disease/healthy categories, with RGB and infrared imagery from real paddy fields. It is a particularly useful future extension for the thermal/IR branch.
