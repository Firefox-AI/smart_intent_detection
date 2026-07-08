# Smart Intent Detection
### This repository is for training intent detection

## Training

### Train a classifier on top of a transformer model
- Example command
`python training/train.py --data-train data/balanced_training_data_v3-2.csv --data-eval data/balanced_evaluation_data_v3-2.csv --base-model csarron/mobilebert-uncased-squad-v2 --model-type mobilebert --output-dir results_mobilebert_v3-2 --epoch 3`

### Train static embedding + transformer encoder + classifier head

#### Step 0 - Fetch data

`./tranining_translations_embeddings/fetch_data.sh`

#### Step 1 - Export the static embedding model

`mkdir -p ./source`

`curl -sSL -o ./tranining_translations_embeddings/source/final.model.npz.best-chrf.npz "https://storage.googleapis.com/moz-fx-translations-data--303e-prod-translations-data/models/en-fr/retrain_hr_NLIxDbE1TBGyOTI-zwZagw/student/final.model.npz.best-chrf.npz"`

`curl -sSL -o ./tranining_translations_embeddings/source/vocab.en.spm "https://storage.googleapis.com/moz-fx-translations-data--303e-prod-translations-data/models/en-fr/retrain_hr_NLIxDbE1TBGyOTI-zwZagw/student/vocab.en.spm"`

`cd ./tranining_translations_embeddings && python export-embeddings-npz.py /tmp/_emb.txt --npz source/final.model.npz.best-chrf.npz --vocab source/enfr-npz/vocab.en.spm --write-model2vec --output-dir ../`

#### Step 2 - create artifacts

- create /yourpathtofx/firefox/artifacts and move export_combined.py, train_tiny_transformer.py build_label_report.py export_fr_bal_onnx.py build_label_report.py export_fr_bal_onnx.py to that directory

#### Step 3 — Build the dataset

`cd /yourpathtofx/firefox/artifacts`

`python export_combined.py /yourpathtofx/firefox/artifacts/intent2`

#### Step 4 — Train the model

`python train_tiny_transformer.py --embeddings scaled --no-pos-weight --fr-oversample 8 --out intent_tiny_fr_oversample --static-model "/your_path_to_smart_intent_directory/tranining_translations_embeddings/en-fr-static-model2vec" --npz "/your_path_to_smart_intent_directory/tranining_translations_embeddings/source/final.model.npz.best-chrf.npz"`

#### Step 5 — Per-label report

`python build_label_report.py`

#### Step 6 — Export to ONNX

`python export_fr_bal_onnx.py`
`python build_label_report.py`

#### Step 7 — Package for transformers.js

`python build_tfjs_package.py`


## Data generation
Use data_generation/generate_balanced_dataset.py

