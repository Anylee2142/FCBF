# Outlines
1. Implementation of paper, Fast Correlation Based Feature Selection (http://www.aaai.org/Papers/ICML/2003/ICML03-111.pdf)
2. Extract optimal feature sets using `target`, `Symmetric Uncertainty`
    > Highly correlated to target  
    > Remove redundant features


```python
from fcbf import fcbf
X = features
y = target
feature_set, history = fcbf(X, y, threshold=0, base=2, is_debug=True)
# `feature_set` refers tooptimal feature set
# `history` contains removed redundant features
```

# Concepts
1. Optimal feature set consists of `Predominant Features`
2. `Predominant Features` are features that
    - (1) highly correlated to target
    - (2) there's no feature that correlated more than (1)
    
# References
- Feature Selection for High-Dimensional Data: A Fast Correlation-Based Filter Solution (http://www.aaai.org/Papers/ICML/2003/ICML03-111.pdf)

TODO
- data to s3
