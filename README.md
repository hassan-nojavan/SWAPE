# SWAPE

Code for [A Nonconvex Framework for Joint Sparse Sample Weighting and Parameter Estimation](https://arxiv.org/abs/2610.33957) by Hassan Nojavan, Chul Moon, and Miju Ahn.

## Settings

The scripts apply SWAPE across three applications. The linear regression example evaluates prediction error and outlier detection on clean data and on data with vertical or interior contamination. The neural network example examines binary classification on two-moons data with flipped training labels under four training configurations (no additional regularization, L2 regularization, dropout, and early stopping). The recommender example uses synthetic ratings, while the paper’s recommender experiments use the Netflix Prize dataset.

## Citation

If you use this code, please cite the paper.

```bibtex
@misc{nojavan2026nonconvexframeworkjointsparse,
      title={A Nonconvex Framework for Joint Sparse Sample Weighting and Parameter Estimation}, 
      author={Hassan Nojavan and Chul Moon and Miju Ahn},
      year={2026},
      eprint={2610.33957},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2610.33957}, 
}
```
