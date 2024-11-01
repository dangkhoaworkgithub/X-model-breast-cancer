# X-Model Transformers

## A Hybrid CNN-ViT Approach for Histopathological Breast Cancer Classification

### Dataset (BreakHis)
- Benign
![collage-0](https://github.com/user-attachments/assets/99a5a00a-b393-45af-8e18-95bc9bc6c228)
- Malignant
![collage-1](https://github.com/user-attachments/assets/d10ea1ac-5a81-4b7b-ac2c-e30f69d4a85d)


### Requirements:

- Pytorch >= 1.4

### Train:
pip install -r `requirements.txt`

To train the model on your own, you can use the following parameters:

- `e`: epoch
- `s`: session - (`g`) - GPU or (`c`) - CPU
- `w`: weight decay (default: 0.0000001)
- `l`: learning rate (default: 0.001)
- `d`: path file
- `b`: batch size (default: 32)
- `p`: The process of accuracy and loss

#### Example command:

To train the model using specific parameters:

```bash
python train.py -e 25 -s 'g' -l 0.0001 -w 0.0000001 -d data/sample_train_data/ -p
```

### Authors:
- Le Dang Khoa

