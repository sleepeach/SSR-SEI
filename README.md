# SSR-SEI
## Paper: M. Tao et al., "Robust Specific Emitter Identification With Sample Selection and Regularization Under Label Noise," in IEEE Internet of Things Journal, doi: 10.1109/JIOT.2024.3453297.[[Paper](https://ieeexplore.ieee.org/document/10663350)][[Code](https://github.com/sleepeach/SSR-SEI)]
## Requirement
pytorch 1.10.2 python 3.6.13
## Dataset
Experiments are conducted on ADS-B and WiFi datasets.

[25] Y. Tu, Y. Lin, H. Zha, J. Zhang, Y. Wang, G. Gui, and S. Mao, “Large scale real-world radio signal recognition with deep learning,” Chinese J. Aeronaut., vol. 35, no. 9, pp. 35–48, Sept. 2022.

[26] K. Sankhe, M. Belgiovine, F. Zhou, S. Riyaz, S. Ioannidis, and K. Chowdhury, “ORACLE: Optimized radio classification through convolutional neural networks,” in IEEE Conference on Computer Communications (ICCC), 2019, pp. 370–378.

## Code introduction
utils-->functions used for training and testing

model-->functions for building model

data_load-->load dataset

optimizer-->functions for optimizer (SGD and APGD)

config-->hyperparameters and paths

loss-->different loss functions

train-->train and test
## E-mail
If you have any question, please feel free to contact us by e-mail (1022010435@njupt.edu.cn).

