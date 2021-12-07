# OOD Detection Papers
> Out-of-Distribution detection papers in taxonomy

## Survey:

- A Unified Survey on Anomaly, Novelty, Open-Set, and Out-of-Distribution Detection: Solutions and Future Challenges. arXiv 2110.
- Generalized Out-of-Distribution Detection: A Survey. arXiv 2110.
- A Unifying Review of Deep and Shallow Anomaly Detection. arXiv 2009; Proceedings of the IEEE 2021.
- Deep Learning for Anomaly Detection: A Review. arXiv 2007; CSUR 2020.
- Deep Learning for Anomaly Detection: A Survey. arXiv 1901.

## Theory:

- Open Category Detection with PAC Guarantees. arXiv 1808; ICML 2018.
- Learning Bounds for Open-set Learning. arXiv 2106; ICML 2021.

## Algorithm:

### Discriminative Models

#### Output-based:

- A baseline for detecting misclassified and out-of-distribution examples in neural networks. arXiv 1610; ICLR 2017.
- Enhancing the reliability of out-of-distribution image detection in neural networks. arXiv 1706; ICLR 2018.

#### Feature-based:

- A simple unified framework for detecting out-of-distribution samples and adversarial attacks. arXiv 1807; NeurIPS 2018.
- Detecting out-of-distribution examples with in-distribution examples and gram matrices. arXiv 1912; ICML 2020.
- Generalized ODIN: Detecting Out-of-distribution Image without Learning from Out-of-distribution Data.arXiv 2002; CVPR 2020.
- Energy-based out-of-distribution detection. arXiv 2010; NeurIPS 2020.
- Convolutional Neural Networks with Compression Complexity Pooling for Out-of-Distribution Image Detection. IJCAI 2020.

#### Ensemble-based:

- Simple and scalable predictive uncertainty estimation using deep ensembles. arXiv 1612; NeurIPS 2017.
- Out-of-distribution detection using an ensemble of self supervised leave-out classifiers. arXiv 1809; ECCV 2018.

#### Auxiliary dataset (select or generate and utilize a set of external OOD data):

- Oepn-Category Classification by Adversarial Sample Generation. arXiv 1705; IJCAI 2017.
- Training confidence-calibrated classifiers for detecting out-of-distribution samples. arXiv 1711; ICLR 2018.
- Novelty Detection with GAN. arXiv 1802.
- Predictive uncertainty estimation via prior networks. arXiv 1802; NeurIPS 2018.
- Discriminative out-of-distribution detection for semantic segmentation. arXiv 1808.
- Reducing network agnostophobia. arXiv 1811; NeurIPS 2018.
- Deep anomaly detection with outlier exposure. arXiv 1812; ICLR 2019.
- Building robust classifiers through generation of confident out of distribution examples. arXiv 1812; NeurIPS-W 2018.
- Out-of-distribution for generalized zero-shot action recognition. arXiv 1904; CVPR 2019.
- Unsupervised out-of-distribution detection by maximum classifier discrepancy. arXiv 1908; ICCV 2019.
- Out-of-distribution detection in classifiers via generation. arXiv 1910; NeurIPS-W 2019.
- Background data resampling for outlier-aware classification. CVPR 2020.
- Bridging In- and Out-of-distribution Samples for Their Better Discriminability. arXiv 2101.
- CODEs: Chamfer Out-of-Distribution Examples against Overconfidence Issue. arXiv 2108; ICCV 2021.
- Semantically Coherent Out-of-Distribution Detection. arXiv 2108; ICCV 2021.

#### Data Augmentation:

- Improved regularization of convolutional neural networks with cutout. arXiv 1708.
- Safer Classification via Synthesis. arXiv 1711.
- On mixup training: Improved calibration and predictive uncertainty for deep neural networks. arXiv 1905; NeurIPS 2019.
- Cutmix: Regularization strategy to train strong classifiers with localizable features. arXiv 1905; CVPR 2019.
- Augmix: A simple data processing method to improve robustness and uncertainty. arXiv 1912.

#### Self-Supervised:

- Deep Anomaly Detection Using Geometric Transformations. arXiv 1805; NIPS 2018.
- Using self-supervised learning can improve model robustness and uncertainty. arXiv 1906; NeurIPS 2019.
- Self-supervised learning for generalizable out-of-distribution detection. AAAI 2020.
- Classification-based anomaly detection for general data. arXiv 2005; ICLR 2020.
- CSI: Novelty detection via contrastive learning on distributionally shifted instances. arXiv 2007; NeurIPS 2020.
- Contrastive training for improved Out-of-distribution Detection. arXiv 2007.
- SSD: A Unified Framework for Self-supervised outlier detection. arXiv 2103; ICLR 2021. 

### Generative Models:

- WAIC, but why? Generative ensembles for robust anomaly detection. arXiv 1810.
- Do deep generative models know what they don't know? arXiv 1810; ICLR 2019.
- Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem. arXiv 1812; CVPR 2019.
- Likelihood Ratios for Out-of-Distribution Detection. arXiv 1906; NeurIPS 2019.
- Towards Neural Network that Provably know when they don't know. arXiv 1909; ICLR 2020.
- Input complexity and out-of-distribution detection with likelihood-based generative models.arXiv 1909; ICLR 2020.
- Likelihood regret: An out-of-distribution detection score for variational auto-encoder. arXiv 2003; NeurIPS 2020.
- Why normalizing flows fail to detect out-of-distribution data. arXiv 2006; NeurIPS 2020.
- Density of States Estimation for Out-of-Distribution Detection. arXiv 2006; AISTATS 2021.

#### Reconstruction: 

> (The GAN will perform better when generating images from previously seen objects (in-distribution data) than objects it has never seen before (out-of-distribution data).

- Finding anomalies with generative adversarial networks for a patrolbot. CVPRW 2017.
- A baseline for detecting misclassified and out-of-distribution examples in neural networks. ICLR 2017.
- Improving Reconstruction Autoencoder Out-of-distribution Detection with Mahalanobis Distance. arXiv 2018.

#### Density-based:

- Deep autoencoding gaussian mixture model for unsupervised anomaly detection. ICLR 2018.
- Latent space autoregression for novelty detection. CVPR 2019.
- Generative probabilistic novelty detection with adversarial autoencoders. NeurIPS 2018.
- Image anomaly detection with generative adversarial networks. ECML&KDD 2018.
- Normalizing flows: An introduction and review of current methods. T-PAMI 2020.
- Adversarially learned one-class classifier for novelty detection. CVPR 2018.
- Glow: Generative flow with invertible 1x1 convolutions. NeurIPS 2018.
- Pixel recurrent neural networks. ICML 2016.

#### Bayesian Models:

- Dropout as a bayesian approximation: Representing model uncertainty in deep learning. ICML 2016.
- Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS 2017.
- Practical deep learning with bayesian principles. NeurIPS 2019.
- Predictive uncertainty estimation via prior networks. NeurIPS 2018.
- Reverse kl-divergence training of prior networks: Improved uncertainty and adversarial robustness. NeurIPS 2019.
- Towards maximizing the representation gap between in-domain & out-of-distribution examples. NeurIPS 2020.

## Settings:

### Semantic-shift:

- Detecting Semantic Anomalies. arXiv 1908; AAAI 2020.
- Transfer-Based Semantic Anomaly Detection. ICML 2021.

### Spurious correlation:

- On the impact of spurious correlation for out-of-distribution detection. arXiv 2109.
- Systematic Generalisation with Group Invariant Predictions. ICLR 2021.

### Near&Far-OOD:

- Does your dermatology classifier know what it doesn't know?detecting the long-tail of unseen conditions.
- Contrastive training for improved out-of-distribution detection.
- A simple fix to Mahalanobis distance for improving Near-OOD detection. ICML workshop uncertainty&robustness 2021.
- Using self-supervised learning can improve model robustness and uncertainty. NeurIPS 2019.

### Large-scale:

- MOS: Towards Scaling Out-of-distribution Detection for Large Semantic Space. CVPR 2021 oral.
- Scaling out-of-distribution detection for real-world settings.

### Fine-grained:

- Fine-grained Out-of-Distribution Detection with Mixup Outlier Exposure. arXiv 2106.

### Multi-label:

- Can multi-label classification networks know what they don't know? arXiv 2109; CVPR 2021.

### Robust Out-of-Distribution Detection:

- Informative Outlier Matters: Robustifying Out-of-distribution Detection Using Outlier Mining. ICML UDL 2020.
- Certifiably adversarially robust detection of out-of-distribution data. NeurIPS 2020.
- Robust out-of-distribution detection for neural networks. arXiv 2020.
- Why relu networks yield high-confidence predictions far away from the training data and how to mitigate the problem. CVPR 2019.
- Novelty detection via blurring. ICLR 2020.
- Atom: Robustifying out-of-distribution detection using outlier mining. ECML&PKDD 2021.
