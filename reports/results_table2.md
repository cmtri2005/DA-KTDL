# Table 2: Classification Evaluation Results

Classification experiments using cluster signals from Phase 4.

## Best Configuration

**Clustering Mode:** abstract
**Classifier Input:** hybrid
**Model:** scibert
**Accuracy:** 0.8140
**Macro F1:** 0.5314
**Top-3 Accuracy:** 0.9630

## Top 10 Results

| Clustering_Mode   | Classifier_Input   | Model   |   Accuracy |   Macro_F1 |   Weighted_F1 |      MCC |   Cohen_Kappa |   Top3_Accuracy |   ROC_AUC_OvR |
|:------------------|:-------------------|:--------|-----------:|-----------:|--------------:|---------:|--------------:|----------------:|--------------:|
| abstract          | hybrid             | scibert |     0.814  |   0.531429 |      0.808904 | 0.769817 |      0.76957  |          0.963  |           nan |
| triples           | abstract           | scibert |     0.8135 |   0.531491 |      0.807159 | 0.768963 |      0.768753 |          0.9645 |           nan |
| hybrid            | abstract           | scibert |     0.8115 |   0.50921  |      0.803868 | 0.766526 |      0.766258 |          0.963  |           nan |
| triples           | concatenate        | scibert |     0.811  |   0.518568 |      0.804595 | 0.765809 |      0.765589 |          0.965  |           nan |
| hybrid            | hybrid             | scibert |     0.8105 |   0.513944 |      0.803871 | 0.765466 |      0.765198 |          0.9625 |           nan |
| concatenate       | concatenate        | scibert |     0.8085 |   0.499292 |      0.800703 | 0.763082 |      0.762811 |          0.9605 |           nan |
| concatenate       | hybrid             | scibert |     0.808  |   0.498242 |      0.800242 | 0.762419 |      0.762162 |          0.9615 |           nan |
| concatenate       | abstract           | scibert |     0.8075 |   0.49576  |      0.799806 | 0.76187  |      0.761591 |          0.9625 |           nan |
| triples           | hybrid             | scibert |     0.8065 |   0.502233 |      0.798733 | 0.760334 |      0.760002 |          0.963  |           nan |
| hybrid            | concatenate        | specter |     0.8055 |   0.503997 |      0.795393 | 0.758846 |      0.758585 |          0.9545 |           nan |