# CropShield AI — Multi-dataset disease training plan

## Primary public datasets
1. PlantVillage — 54,303 leaf images, 38 species/disease classes. TensorFlow Datasets documents the dataset and its 38 classes.
2. Paddy Doctor — 16,225 annotated paddy leaf images, 13 classes (12 diseases + normal).
3. Rice Leaf Disease (multi-source) — 30,000+ field/lab images, 17 disease classes plus healthy/pest, MIT license according to the public dataset card.
4. Rice Leaf Diseases — 120 images, 3 classes (leaf smut, brown spot, bacterial leaf blight). Useful only as a small external validation set.
5. Rice disease datasets with 6–8 classes can be added as external validation sets; keep train/validation sources separated to reduce leakage.

## Recommended training policy
- Train the generic multi-crop classifier primarily on PlantVillage.
- Train/benchmark a rice-specific model using Paddy Doctor and the larger rice dataset.
- Keep external datasets as a held-out validation/test pool where possible.
- Do not merge duplicate/augmented images across sources before splitting.
- Report macro F1, per-class recall and confusion matrix, not only accuracy.

## Important
Large image datasets are intentionally not embedded in this ZIP. The repository contains download/training scripts because bundling tens of thousands of images would make the project unnecessarily huge and can create redistribution/licensing problems.
