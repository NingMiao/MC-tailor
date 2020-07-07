# MC-tailor
Codes for <Do You Have the Right Scissors? Tailoring Pre-trained Language Models via Monte-Carlo Methods>
---
###Introduction
MC-tailor is a Monte-Carlo based method aimed to improve language-model finetuning. It consists of a ratio estimator and a sampler. Please refer to our paper for details.

###Dependencies
`python 3.7`
`tensorflow 1.13` Please download GPT-2 small (117M) and put it under `./model` 
`toposort`

###Datasets
All datasets used in our paper are attached here. Please don't forget to cite corresponding papers if you are going to use them.

###Usage
-Firstly, perform finetuning (as baseline and the base model for tailoring)
 `python3 Tailor_SMC.py/Tailor_ERS.py --finetune`
-To evaluate finetuning model
 `python3 Tailor_SMC.py/Tailor_ERS.py --evaluate_finetune`
-To perform tailoring
 `python3 Tailor_SMC.py/Tailor_ERS.py --train_tailor`
-To evaluate the tailor
 `python3 Tailor_SMC.py/Tailor_ERS.py --evaluate_tailor`
