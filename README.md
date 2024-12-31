# Credit Fraud Detection end to end Machine Learning Project

## Environment setup

- Create new environment
    ``` 
    conda create -p venv python==3.8 -y 
    ```
- Run the environment
    ```
    conda activate venv/
    ```

## Git
- Revert recent commit
    ```
    git reset --soft HEAD~1
    ```
- Add heavy file to .gitattributes
    ```
    git lfs track "path/to/file"
    ```


## 2. Data Transformation (e.g., Scaling, Encoding) After Splitting
Fit Transformations on Training Data Only:

Fit: Calculate the necessary parameters (e.g., mean and standard deviation for scaling) using only the training data.
Transform: Apply these parameters to both the training and test sets. This ensures that the test data is scaled based on the training data's distribution, maintaining its role as unseen data.
Avoid Fitting on Test Data:

Why? Including test data in the fitting process can introduce data leakage, where the model gains unintended insights into the test set, compromising its ability to generalize.