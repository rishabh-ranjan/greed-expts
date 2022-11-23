# GREED Experiments

This repository contains the Jupyter notebooks used to run the experiments reported in the paper ["GREED: A Neural Framework for Learning Graph Distance Functions"](https://arxiv.org/abs/2112.13143) accepted at NeurIPS 2022. The main repository containing source code is [here](https://github.com/idea-iitd/greed). The notebooks have paths and configurations which are custom to our setup, but these can easily be adapted to experiment in general settings.

## Directory descriptions

1. `nbs_train`: train various models

2. `nbs_pred`: make predictions using the trained models

3. `nbs_regress`: regression experiments

4. `nbs_rank`: ranking experiments

5. `nbs_range`: range query experiments

6. `nbs_index`: retrieval experiments using index structures

## Citation

```
@inproceedings{ranjan&al22,
  author = {Ranjan, Rishabh and Grover, Siddharth and Medya, Sourav and Chakaravarthy, Venkatesan and Sabharwal, Yogish and Ranu, Sayan},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {GREED: A Neural Framework for Learning Graph Distance Functions},
  booktitle = {Advances in Neural Information Processing Systems 36: Annual Conference
               on Neural Information Processing Systems 2022, NeurIPS 2022, November 29-Decemer 1, 2022},
  year = {2022},
}
```

