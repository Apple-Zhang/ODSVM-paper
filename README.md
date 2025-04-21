# ODSVM-paper
Implementation of "Learning the Optimal Discriminant SVM With Feature Extraction". The paper is published in IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), see [here](https://doi:10.1109/TPAMI.2025.3529711).

To run the demo, go to `./matlab` and run
```[matlab]
>>> demo
```
under Matlab environment.

The main files are as follows:
- `bODSVM_train`: The training process of binary-class ODSVM.
- `bODSVM_predict`: The prediction process of binary-class ODSVM.
- `mODSVM_train`: The training process of multi-class ODSVM. 
- `mODSVM_predict`: The prediction process of multi-class ODSVM.

The codes depends on the matlab interface of [LIBLINEAR](https://github.com/cjlin1/liblinear):
- `liblineartrain`: Training of linear SVM
- `liblinearpredict`: Inference of linear SVM.

This repo includes the compiled mex files under windows64.
If you cannot run them, try to recompile them from source following the instructions of [LIBLINEAR-Matlab](https://github.com/cjlin1/liblinear/tree/master/matlab).

Besides, if you use "csmo" option to run `mODSVM_train`, you should first ensure that you can run `Csubqp_smo` subroutine.
To recompile it, go to `./matlab/mexfile` and run 
```[matlab]
>>> mex Csubqp_smo
```
under Matlab environment.
`csmo` option is usually faster than `matlab-smo` option.

If you use this code or find our work helpful, please cite this paper through:
```[LaTeX]
@article{ZLKY2025,
  author={Zhang, Junhong and Lai, Zhihui and Kong, Heng and Yang, Jian},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Learning the Optimal Discriminant SVM With Feature Extraction}, 
  year={2025},
  volume={47},
  number={4},
  pages={2897-2911},
  keywords={Support vector machines;Feature extraction;Optimization;Classification algorithms;Convergence;Vectors;Principal component analysis;Minimization;Representation learning;Training;Support vector machine;subspace learning;joint learning framework},
  doi={10.1109/TPAMI.2025.3529711}
}
```
