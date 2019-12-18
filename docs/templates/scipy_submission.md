# Hyperopt: A Python library for optimizing the hyperparameters of machine learning algorithms

[SciPy2013 Abstract submission](http://conference.scipy.org/scipy2013/speaking_submission.php)

## Authors

James Bergstra, Dan Yamins, and David D. Cox

## Bios

James Bergstra is an NSERC Banting Fellow at the University of Waterloo's Centre for Theoretical Neuroscience.
His research interests include visual system models and learning algorithms, deep learning, Bayesian optimization, high performance computing, and music information retrieval.
Previously he was a member of Professor David Cox's Computer and Biological Vision Lab in the Rowland Institute for Science at Harvard University.
He completed doctoral studies at the University of Montreal in July 2011 under the direction of Professor Yoshua Bengio with a dissertation on how to incorporate complex cells into deep learning models.
As part of his doctoral work he co-developed Theano, a popular meta-programming system for Python that can target GPUs for high-performance computation.

Dan Yamins is a post-doctoral research fellow in Brain and Cognitive Sciences at the Massachusetts Institute of Technology.  His research interests include computational models of the ventral visual stream, and high-performance computing for neuroscience and computer vision applications.  Previously, he developed python-language software tools for large-scale data analysis and workflow management.  He completed his PhD at Harvard University under the direction of Radhika Nagpal, with a dissertation on computational models of spatially distributed multi-agent systems.

David Cox is an Assistant Professor of Molecular and Cellular Biology and of Computer Science, and is a member of the Center for Brain Science at Harvard University. He completed his Ph.D. in the Department of Brain and Cognitive Sciences at MIT with a specialization in computational neuroscience. Prior to joining MCB/CBS, he was a Junior Fellow at the Rowland Institute at Harvard, a multidisciplinary institute focused on high-risk, high-reward scientific research at the boundaries of traditional fields.

## Talk Summary

Most machine learning algorithms have hyperparameters that have a great impact on end-to-end system performance, and adjusting hyperparameters to optimize end-to-end performance can be a daunting task.
Hyperparameters come in many varieties--continuous-valued ones with and without bounds, discrete ones that are either ordered or not, and conditional ones that do not even always apply
(e.g., the parameters of an optional pre-processing stage)--so
conventional continuous and combinatorial optimization algorithms either do not directly apply, or else operate without leveraging structure in the search space.
Typically, the optimization of hyperparameters is carried out before-hand by  domain experts on unrelated problems, or manually for the problem at hand with the assistance of grid search.
However, when dealing with more than a few hyperparameters (e.g. 5), the standard practice of manual search with grid refinement is so inefficient that even random search has been shown to be competitive with domain experts [1].

There is a strong need for better hyperparameter optimization algorithms (HOAs) for two reasons:

1. HOAs formalize the practice of model evaluation, so that benchmarking experiments can be reproduced at later dates, and by different people.

2. Learning algorithm designers can deliver flexible fully-configurable implementations to non-experts (e.g. deep learning systems), so long as they also provide a corresponding HOA.

Hyperopt provides serial and parallelizable HOAs via a Python library [2, 3].
Fundamental to its design is a protocol for communication between
(a) the description of a hyperparameter search space,
(b) a hyperparameter evaluation function (machine learning system), and
(c) a hyperparameter search algorithm.
This protocol makes it possible to make generic HOAs (such as the bundled "TPE" algorithm) work for a range of specific search problems.
Specific machine learning algorithms (or algorithm families) are implemented as hyperopt _search spaces_ in related projects:
Deep Belief Networks [4],
convolutional vision architectures [5],
and scikit-learn classifiers [6].
My presentation will explain what problem hyperopt solves, how to use it, and how it can deliver accurate models from data alone, without operator intervention.

## Submission References

[1] J. Bergstra and Y. Bengio (2012).  Random Search for Hyper-Parameter Optimization.  Journal of Machine Learning Research 13:281–305.
http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf

[2] J. Bergstra, D. Yamins and D. D. Cox (2013).  Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures.  Proc. 30th International Conference on Machine Learning (ICML-13).
http://jmlr.csail.mit.edu/proceedings/papers/v28/bergstra13.pdf

[3] Hyperopt: http://hyperopt.github.com/hyperopt

[4] ... for Deep Belief Networks: https://github.com/hyperopt/hyperopt-nnet

[5] ... for convolutional vision architectures: https://github.com/hyperopt/hyperopt-convnet

[6] ... for scikit-learn classifiers: https://github.com/hyperopt/hyperopt-sklearn

More information about the presenting author can be found on his academic website: http://www.eng.uwaterloo.ca/~jbergstr/
