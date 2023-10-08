# Glioma Data Analysis
Repository for competition submission

## How to start
Get the competition data from [Kaggle](https://www.kaggle.com/competitions/60504). And place it into `datasets` folder.

Prerequirements: `Python: 3.9+`

```bash
pip install -r requirements.txt
export PYTHONPATH=${PYTHONPATH}:$(pwd)
```

## Task description

- [x] 1. Your final result should overcome the F1=0.5. Any team with lower results is failed (all or nothing) 
- [x] 2. You must provide your working pipeline on the last day of submission. Please, make it look well 
- [x] 3. Base EDA and data preprocessing: (+3)

    - [x] a. clarify the meaning of features you are going to exploit

    - [x] b. Execute the correlation analysis of variables, try to establish dependent ones

    - [x] c. Perform feature selection

    - [x] d. Perform data preprocessing (encode your features in some way, different types of encoding may dramatically affect your model, be selective, try multiple (at least 2) ways)
    
- [x] 4. Try at least 3 different types of models (+2)
- [ ] 5. Try an ensemble model with trainable and non-trainable weights. (+2)
- [x] 6. Make explainable model (+2)
- [ ] 7. Aggregate the results, proceed the error analysis, compare the models (+1)
- [ ] 8. *Try to get the highest score (Top 1 +3; Top 2 +2; Top 3 +1)
- [ ] 9. *Try NN approach for the task (+1)
- [ ] 10. *Make explainable model with top score (+2)
