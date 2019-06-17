###End to End Speech Recognize in Chinese
##### Under progress

##### Model: 
    Transformer:
##### File System:

    --ckpt
    --data
        --data_collector: build data manifist
        --data_loader :dataset & dataloader
        extract_data.py: extract .tar file
    --Predictor
        --bases : base model, base configer
        --data_handler
            --vocab: vocabulary
            --data_parser: data process pipeline
        --Models: models
        --Utils
            --loss
            --score
        predictor.py
    --Trainer
        --trainer
        --optimizer
    --main.py
    --preprocess.py

#####usage:
    
    put datafile in data/
    custom data_extract.py
    custom data_collector
    custom data_loader
    
    python preprocess.py
    python main.py --num_epoch=20
    
