このリポジトリは、[LowProFool](https://github.com/axa-rev-research/LowProFool)のリポジトリを再現するためにDockerで再現するためにしたものです。  
This repository is my hands-on repository from LowProFool by Github[LowProFool](https://github.com/axa-rev-research/LowProFool).

__Disclaimer:__ This repository is not maintained anymore

------

# LowProFool

LowProFool is an algorithm that generates imperceptible adversarial examples

This GitHub hosts the code to replicate the experiments presented in the paper:

Ballet, V., Renard, X., Aigrain, J., Laugel, T., Frossard, P., & Detyniecki, M. (2019). Imperceptible Adversarial Attacks on Tabular Data. NeurIPS 2019 Workshop on Robust AI in Financial Services: Data, Fairness, Explainability, Trustworthiness, and Privacy (Robust AI in FS 2019)

https://arxiv.org/abs/1911.03274

### Adverse.py

Contains the implementation of LowProFool [[1]](about:blank) along with an modifier version of DeepFool [[2]](about:blank) that handles tabular datasets.

### Metrics.py

Implements metrics introduced in [[1]](about:blank)

### Playground.ipynb

A demo python notebook to generate adversarial examples on the German Credit dataset and compare the results to DeepFool

## References
[1] Ballet, V., Renard, X., Aigrain, J., Laugel, T., Frossard, P., & Detyniecki, M. (2019). Imperceptible Adversarial Attacks on Tabular Data. NeurIPS 2019 Workshop on Robust AI in Financial Services: Data, Fairness, Explainability, Trustworthiness, and Privacy (Robust AI in FS 2019)

[2] S. Moosavi-Dezfooli, A. Fawzi, P. Frossard: DeepFool: a simple and accurate method to fool deep neural networks. In Computer Vision and Pattern Recognition (CVPR ’16), IEEE, 2016.

---

## データセットの中身

| checking_status | duration | credit_amount | savings_status | employment | installment_commitment | residence_since | age | existing_credits | num_dependents | own_telephone | foreign_worker | target |
|-----------------|----------|---------------|----------------|------------|------------------------|-----------------|-----|-------------------|----------------|---------------|----------------|--------|
| <0              | 6.0      | 1169.0        | no known savings | >=7      | 4.0                    | 4.0             | 67.0| 2.0               | 1.0            | yes           | yes            | 1.0    |
| 0<=X<200        | 48.0     | 5951.0        | <100           | 1<=X<4    | 2.0                    | 2.0             | 22.0| 1.0               | 1.0            | none          | yes            | 0.0    |
| no checking     | 12.0     | 2096.0        | <100           | 4<=X<7    | 2.0                    | 3.0             | 49.0| 1.0               | 2.0            | none          | yes            | 1.0    |
| <0              | 42.0     | 7882.0        | <100           | 4<=X<7    | 2.0                    | 4.0             | 45.0| 1.0               | 2.0            | none          | yes            | 1.0    |
| <0              | 24.0     | 4870.0        | <100           | 1<=X<4    | 3.0                    | 4.0             | 53.0| 2.0               | 2.0            | none          | yes            | 0.0    |