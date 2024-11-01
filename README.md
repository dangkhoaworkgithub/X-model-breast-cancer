# X-Model Transformers

## A Hybrid CNN-ViT Approach for Histopathological Breast Cancer Classification

### Dataset (BreakHis)
<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/99a5a00a-b393-45af-8e18-95bc9bc6c228" alt="Benign" width="200"/>
      <p align="center">Benign</p>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/d10ea1ac-5a81-4b7b-ac2c-e30f69d4a85d" alt="Malignant" width="200"/>
      <p align="center">Malignant</p>
    </td>
  </tr>
</table>

### Model Architecture

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/883be3e3-a5b9-4a1b-8477-cd3b66c81d41" alt="XModel Architecture" width="800"/>
      <p>Model Architecture</p>
    </td>
  </tr>
</table>


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

