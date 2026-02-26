# This repository is for training intent detection

# Training
- Example command
`python training/train.py --data-train data/balanced_training_data_v3-2.csv --data-eval data/balanced_evaluation_data_v3-2.csv --base-model csarron/mobilebert-uncased-squad-v2 --model-type mobilebert --output-dir results_mobilebert_v3-2 --epoch 3`

# Data generation
Use data_generation/generate_balanced_dataset.py

