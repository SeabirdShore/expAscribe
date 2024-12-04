# Team ZJU-China 2024 Software Tool
                          
![](https://tinypic.host/images/2024/09/18/_2024-08-30_042234-removebg-preview.png)
![Static Badge](https://img.shields.io/badge/expAscribe-1.2.1-green)
![](https://badgen.net/static/Python/3.10/blue)
![Static Badge](https://img.shields.io/badge/LICENSE-CC_BY_4.0-blue)


ExpAscribe: a causality analysis framework with causal graph discovery, user-given graph falsification and threshold analysis based on interaction terms.

## Table of Contents
<p style="text-align:center">
  <a href="#description">Description</a> &#xa0; | &#xa0;
  <a href="#installation">Installation</a> &#xa0; | &#xa0;
  <a href="#usage">Usage</a> &#xa0; | &#xa0;
  <a href="#contributing">Contributing</a> &#xa0; | &#xa0;
  <a href="#contributors">Contributors</a>  

## Description

### Overview 
- ExpAscribe: main work of the project, presented in the the form of Python library and WebApp instance. The relevant primary project code, files, and directories include -- `expAscribe`, `setup.py`, `MANIFEST.in`, `notebooks`, `web`. 

For more information, please visit our [Wiki](https://2024.igem.wiki/zju-china/software).

## Installation

```shell
pip install expAscribe
```

If you encounter dependent version conflicts during the installation process.
You can try to create a virtual environment and install it.

```shell
pip install virtualenv
virtualenv <your_env>
source <your_env>/bin/activate # Linux/macOS
<your_env>\Scripts\activate  # Windows
pip install expAscribe
```

Since the installation of this way requires conda, please install [Anaconda](https://www.anaconda.com/download) first.

```shell
conda create -n <your_env> python=3.10
conda activate <your_env>
pip install expAscribe
```

## Usage

For more information on Usage, please visit our [wiki](https://2024.igem.wiki/zju-china/software/) which holds technical articles and tutorial videos, [docs](https://seabirdshore.github.io/EAdocs/) which holds API documentation and example notebooks.

## Contributing

We greatly appreciate and encourage contributions to this project. To ensure a smooth process, please follow the steps below when submitting your pull requests.

### Pull Requests

1. Fork the repository and create a new branch from the `main` branch.
2. If you’ve added new functionality, ensure you include appropriate tests.
3. For any changes to the API, be sure to update the documentation accordingly.
4. Ensure that your code passes all linting checks before submission.

Once you’ve completed these steps, feel free to submit your pull request. If you'd like to discuss any details or require assistance, you can reach me via 3220105728 at zju dot edu dot cn.

## Contributors

### Author

- Xiao Sibo, dev and algorithm design @College of Computer Science and Technology, Zhejiang University

### Acknowledgments

I would like to express my deepest gratitude to the following individuals and organizations for their invaluable contributions...
- He Yichen, Liu Haoyang and Fu Jinyuan, bioinformatics design, experimental verification and data gathering
- Kun Kuang, professor of the Department of Artificial Intelligence, Zhejiang University

<br>

<a href="#top">Back to top</a>

