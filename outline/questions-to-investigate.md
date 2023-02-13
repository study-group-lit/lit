## Task

Fine-tune a PLM (BERT, RoBERTa, BART, ...) for NLI
 - using SNLI, MultiNLI, SICK, ... data sets, plus:
 - using extended datasets (recast for NLI, cf. Lecture)

## Questions

### Model Analysis
 - Does system analysis of fine-tuned NLI models confirm that your model makes correct predictions for the right reasons?
    - Apply interpretability methods
    - Are good answers based on relevant tokens? What does model inspection show? Does it fit your intuitions?
 - Does model analysis confirm that you get improvements for the right reason?

### Suitable for Hypotheses 
 - Why do PLMs not perform well on NLI zero-shot testing?
 - How to make use of data from WordNet, FrameNet or VerbNet to improve NLI?
 - Does fine-tuning on data from linguistic resources improve performance?

## Possible Hypotheses
### Why do PLMs not perform well on NLI zero-shot testing?
[Finetuned Language Models are Zero-Shot Learners (2022)](https://arxiv.org/pdf/2109.01652.pdf)

Introduces FLAN, a PLM additionally trained on instruction templates for different tasks. FLAN produces okay results even in some NLI tasks (CB, RTE)

[What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization? (2022)](https://proceedings.mlr.press/v162/wang22u/wang22u.pdf)

Inspects many different variations of pretraining in architecture and training objectives in order to find the best setup for Zero-Shot Learning.

[Pre-trained Language Models can be Fully Zero-Shot Learners](https://arxiv.org/pdf/2212.06950.pdf)

Propses NPPrompt, a method for fully zero-shot learning with pre-trained language models. Uses initial word embedding of PLM to automatically find related words for category names, which enables them to construct the verbalizers without manual design or unlabeled corpus. Experimental results show that NPPrompt outperforms the previous zero-shot methods but still not very good on NLI tasks.

**Conclusion**: Many approaches to make Zero-Shot results better but no real explanation why results are not good for NLI tasks.

### How to make use of data from WordNet, FrameNet or VerbNet to improve NLI?
[Breaking NLI Systems with Sentences that Require Simple Lexical Inferences (2018)](https://aclanthology.org/P18-2103/)

Generate premise hypothesis pairs by replacing single words in the premise with others also present in the SNLI data set which was used as training data for the systems under inspection. E.g.to generate entailment example replace word with synonym or hypernym. Generated pairs are verified by crowd-source workers. Result: In all tested models drop of accuracy between 11 and 33 points

[Neural Natural Language Inference Models Enhanced with External Knowledge (2018)](https://arxiv.org/pdf/1711.04289.pdf)
Uses data from WordNet and TransE to enhance their system. Result is named KIM, referenced in many other papers. Resulting accuracy: SNLI: 88.6% Glockner's: 83.5%

[Improving Natural Language Inference Using External Knowledge in the Science Questions Domain (2018)](https://ojs.aaai.org/index.php/AAAI/article/view/4705)

System (ConSeqNet) with two parts: (a) Text based: takes as input premise and hypothesis (b) Graph based: takes as input specific knowledge drived from the knowledge base using premise and hypothesis, both results then put through a classifier. Use ConcpetNet because reasons and measurements. Result: 85.2% accuracy.

**Conclusion**: Interesting field with many possibilities but hard and already many papers.

### Does fine-tuning on data from linguistic resources improve performance?
[Fake News Detection as Natural Language Inference](https://arxiv.org/pdf/1907.07347.pdf)

[Improving BERT Fine-Tuning via Self-Ensemble and Self-Distillation](https://arxiv.org/pdf/2002.10345.pdf)
Propeses enhanced fine-tuning, effect seems to be rather low. Accuracy: 90% -> 91%

[oLMpics - On what Language Model Pre-training Captures](https://arxiv.org/pdf/1912.13283.pdf)

**Conclusion**: Fine tuning does not seem to have a very large effect. Mostly ~1%
