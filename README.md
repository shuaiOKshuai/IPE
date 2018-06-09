# Interactive Paths Embedding for Semantic Proximity Search on Heterogeneous Graphs
The resources are for our KDD-18 paper IPE : <br>
Zemin Liu, Vincent W. Zheng, Zhou Zhao, Zhao Li, Hongxia Yang, Minghui Wu and Jing Ying. *Interactive Paths Embedding for Semantic Proximity Search on Heterogeneous Graphs.* In Proc. of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '18), London, United Kingdom, 19 – 23 AUGUST, 2018.

This python project implements the IPE model proposed in the above paper. <br>
Please refer to the above paper for all the details of this model. <br>
If you use it for scientific experiments, please cite this paper:

@inproceedings{LiuZKDD18, <br>
author = {Zemin Liu and <br>
Vincent W. Zheng and <br>
Zhou Zhao and <br>
Zhao Li and <br>
Hongxia Yang and <br>
Minghui Wu and <br>
Jing Ying}, <br>
title = {Interactive Paths Embedding for Semantic Proximity Search on Heterogeneous Graphs}, <br>
booktitle = {The 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '18)}, <br>
year = {2018} <br>
} 

=====================================================================

We realize IPE model in two modes, the serial mode and the batch mode. These two models are written in python, with theano framework. The batch mode could leverage the GPU for parallel processing better. If you only run a small dataset on CPU, you can also use serial mode. We take the serial mode for example.

**pythonParamsConfig** : is the config file. You can set your parameters in this file. <br>
**prepareSubpathsToInteractiveSeqs.py** : is the part for interactive paths generation. Before running the whole model, you should prepare the interactive paths structures by this file. <br>
**experimentForOneFileByParams.py** : is the entry of the model. You can train and test your model by execute this file.

For batch mode, there are another two files: <br>
**SplitTestDatasetsIntoMultiFiles.py** : As there may be too many sequences(interactive paths structures) for the test data, you can filter the valid sequences out for test data, to save space. <br>
**reorgnizeSubpaths.py** : As there may be too many subpaths between some node pairs, we need to filter the subpaths with max path number.

For more details, please refer to our paper. Thanks !
