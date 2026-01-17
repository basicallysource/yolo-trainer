# YOLO Segmentation Fine-tuning

## Config

Copy `config_classification_chamber_example.json` and edit paths:
```json
{
    "data_yaml_path": "your_data.yaml",
    "data_path": "/path/to/dataset",
    "model_size": "small",
    "epochs": 100,
    "batch_size": 20,
    "device": "cuda",
    "val_split": 0.1
}
```

## Local Training

```bash
python finetune_segmenting.py config.json
```

## Remote Training

```bash
# on local
# this will sync the dataset and training code to the ssh'd gpu instance
python sync.py push config.json --remote user@host   # sync code + data

# on the actual machine
python finetune_segmenting.py config.json

# on local
python sync.py pull config.json --remote user@host   # pull checkpoints back
```

## Test

```bash
python test.py checkpoint.pt --webcam 0
```
