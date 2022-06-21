# Conditional-GAT-ECPE
This is a repository for our 2022 WWW paper "Graph Attention Network based Detection of Causality for Textual Emotion-Cause Pair" \[[pdf](https://www.aclweb.org/anthology/2020.emnlp-main.252.pdf)\].

If you use our code, please cite our paper:
>Graph Attention Network based Detection of Causality for Textual Emotion-Cause Pair[J]. World Wide Web: Internet and Web Information Systems(WWW), 2022.

Hardware Environment
- Centos7
- tesla v100

Dependency Requirement
- Python 3.6
- Tensorflow 1.14.0
- sklearn, numpy, scipy

Dataset Construction Steps
- Run the “preprocess.cy” to get the manually labeled dataset, which will be stored in a file called “data.txt”
- Run the “gen_nega_samples.py” to generate the constructed conditional-ECPE dataset, which is stored in a file called “data_wneg.txt”
-	If you prefer our training/testing split, please run “divide_fold.py” to get 20 files, which will be named as “foldx_train.txt” and “foldx_test.txt”, where “x” should be from 1 to 10.

Note that if you directly download the repo zip file from the github site, **the downloaded "w2v_200.txt" in directory "nega_data" may not be the correct file.** Please:
- open the "w2v_200.txt" file in github;
- right click on the website;
- choose "save as" to download the correct file, which should be around **80Mb**. 

If you are cloning the whole repo, the above issue should not be a problem.

To run a program:
- Make sure you complete the dataset construction first
-	Directly run “python programname.py”, where the “programename” is the python file you want to run.

Should you have any problem, contact 483412504@qq.com
