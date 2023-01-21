# Interactive Paths Embedding for Semantic Proximity Search on Heterogeneous Graphs
The resources are for our KDD-18 paper IPE : <br>
Zemin Liu, Vincent W. Zheng, Zhou Zhao, Zhao Li, Hongxia Yang, Minghui Wu and Jing Ying. *Interactive Paths Embedding for Semantic Proximity Search on Heterogeneous Graphs.* In Proc. of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '18), London, United Kingdom, 19 – 23 AUGUST, 2018.

This python project implements the IPE model proposed in the above paper. <br>
Please refer to the above paper for all the details of this model. <br>
If you use it for scientific experiments, please cite this paper:

# Cite

      @inproceedings{LiuZKDD18, 
        author = {Zemin Liu and 
        Vincent W. Zheng and 
        Zhou Zhao and 
        Zhao Li and 
        Hongxia Yang and 
        Minghui Wu and 
        Jing Ying}, 
        title = {Interactive Paths Embedding for Semantic Proximity Search on Heterogeneous Graphs},
        booktitle = {The 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '18)}, 
        year = {2018} 
      } 

=====================================================================

# Details

We realize IPE model in two modes, the serial mode and the batch mode. These two models are written in python, with theano framework. The batch mode could leverage the GPU for parallel processing better. If you only run a small dataset on CPU, you can also use serial mode. We take the serial mode for example.

**pythonParamsConfig** : is the config file. You can set your parameters in this file. <br>
**prepareSubpathsToInteractiveSeqs.py** : is the part for interactive paths generation. Before running the whole model, you should prepare the interactive paths structures by this file. <br>
**experimentForOneFileByParams.py** : is the entry of the model. You can train and test your model by execute this file.

For batch mode, there are another two files: <br>
**SplitTestDatasetsIntoMultiFiles.py** : As there may be too many sequences(interactive paths structures) for the test data, you can filter the valid sequences out for test data, to save space. <br>
**reorgnizeSubpaths.py** : As there may be too many subpaths between some node pairs, we need to filter the subpaths with max path number.

# Note
We also provided the sub-path generation code in one of our previous projects. Please see our project ProxEmbed (https://github.com/shuaiOKshuai/ProxEmbed; in the code, either symmetric or asymmetric is OK; we utilize the java code to prepare the data). In particular, in file https://github.com/shuaiOKshuai/ProxEmbed/blob/master/code/asymmetric/java%20-%20prepare%20data%20for%20model/Main.java, we conduct random walk to sample paths in line 23, and generate the sub-paths from the sampled paths in line 32.

For more details, please refer to our paper. Thanks !
