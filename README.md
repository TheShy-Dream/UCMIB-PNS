# UCMIB-PNS
The official implementation of UCMIB-PNS: Balancing Sufficiency and Necessity with Probabilistic Causality and Cross-Modal Uncertainty in Multimodal Sentiment Analysis. 

## Datasets

You can download the CMU-MOSI and CMU-MOSEI datasets using [CMU-MultimodalDataSDK](https://github.com/Jie-Xie/CMU-MultimodalDataSDK).

You can download the UR-FUNNY dataset using [UR-FUNNY resp](https://github.com/ROC-HCI/UR-FUNNY).

You can find the download link of CH-SIMS dataset in [CH-SIMS](https://aclanthology.org/2020.acl-main.343.pdf).

## Training

```
python main.py --dataset mosi
```

## Environment Requirements

python == 3.8.8

torch == 1.8.1

numpy == 1.20.0

## Citation

If you find this repository useful, please consider citing our paper:

```
@article{chen2025ucmib,
  title={UCMIB-PNS: Balancing Sufficiency and Necessity with Probabilistic Causality and Cross-Modal Uncertainty in Multimodal Sentiment Analysis},
  author={Chen, Jili and Zhong, Yihua and Huang, Qionghao and Huang, Changqin and Jiang, Fan and Huang, Xiaodi and Wang, Xun},
  journal={IEEE Transactions on Affective Computing},
  number={01},
  pages={1--16},
  year={2025},
  publisher={IEEE Computer Society}
}
```