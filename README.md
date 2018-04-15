# SNM: Stochastic Newton Method for Optimization of Discrete Choice Models

This repository contains all the code used to produce the figures and tables for the article **SNM: Stochastic Newton Method for Optimization of Discrete Choice Models**. This article has been sent for reviews at IEEE ITSC 2018. Feel free to use any part of the code at the condition that our article is explicitly  mentionned, see section [Authorization and Reference](#authorization-and-reference). 

**The article is available on this Github repository: [LedLurBie_IEEE2018.pdf](https://github.com/glederrey/IEEE2018_SNM/blob/master/paper/LedLurBie_IEEE2018.pdf).**

If you have any questions about the code or the results, feel free to contact [Gael Lederrey](mailto:gael.lederrey@epfl.ch).

## Content of the repository

This repository contains all the figures used in this article in the folder [`figures`](https://github.com/glederrey/IEEE2018_SNM/tree/master/figures). You can also find the LaTeX for the article in the folder [`paper`](https://github.com/glederrey/IEEE2018_SNM/tree/master/paper). In the folder [`data`](https://github.com/glederrey/IEEE2018_SNM/tree/master/data), we added the original Swissmetro data as well as the data generated from the algorithms used in our article. Finally, in the folder [`code`](https://github.com/glederrey/IEEE2018_SNM/tree/master/code), you can find the following folders:
- [`biogeme`](https://github.com/glederrey/IEEE2018_SNM/tree/master/code/biogeme): it contains the code to run biogeme on the model tested in this article.
- [`classes`](https://github.com/glederrey/IEEE2018_SNM/tree/master/code/classes): it contains the class MNLogit build for this article. It's a simple Multinomial logit model.
- [`helpers`](https://github.com/glederrey/IEEE2018_SNM/tree/master/code/helpers): it contains a bunch of helpers files and functions. **The SNM algorithm is in the file [`code/helpers/algos.py`](https://github.com/glederrey/IEEE2018_SNM/blob/master/code/helpers/algos.py).**
- [`notebooks`](https://github.com/glederrey/IEEE2018_SNM/tree/master/code/notebooks): it contains all the iPython notebooks that were used to generate the figures and tables.

We present now the content of all the iPython notebooks:
- [`norm_vs_raw.ipynb`](https://github.com/glederrey/IEEE2018_SNM/blob/master/code/notebooks/norm_vs_raw.ipynb): This notebook is used for the tests between the raw data and normalized model. Fig. [1(a)](https://github.com/glederrey/IEEE2018_SNM/blob/master/figures/SGD_norm_raw.pdf), [1(b)](https://github.com/glederrey/IEEE2018_SNM/blob/master/figures/adagrad_norm_raw.pdf), and [1(c)](https://github.com/glederrey/IEEE2018_SNM/blob/master/figures/SNM_norm_raw.pdf) are generated in this notebook. Table II is also generated in this notebook.
- [`SGD.ipynb`](https://github.com/glederrey/IEEE2018_SNM/blob/master/code/notebooks/SGD.ipynb): This notebook is used for the benchmark of the first-order methods. Fig. [2(a)](https://github.com/glederrey/IEEE2018_SNM/blob/master/figures/SGD.pdf) is generated in this notebook as well as part of Table III.
- [`SBFGS.ipynb`](https://github.com/glederrey/IEEE2018_SNM/blob/master/code/notebooks/SBFGS.ipynb): This notebook is used for the benchmark of the quasi-newton methods. Fig. [2(b)](https://github.com/glederrey/IEEE2018_SNM/blob/master/figures/SBFGS.pdf) is generated in this notebook as well as part of Table III.
- [`SNM.ipynb`](https://github.com/glederrey/IEEE2018_SNM/blob/master/code/notebooks/SNM.ipynb): This notebook is used for the benchmark of the second-order methods. Fig. [2(c)](https://github.com/glederrey/IEEE2018_SNM/blob/master/figures/SNM.pdf) is generated in this notebook as well as part of Table III.
- [`batch_size.ipynb`](https://github.com/glederrey/IEEE2018_SNM/blob/master/code/notebooks/batch_size.ipynb): This notebook is used to study the effect of the batch size on SNM. Fig. [3](https://github.com/glederrey/IEEE2018_SNM/blob/master/figures/perc_newton.pdf) and [4](https://github.com/glederrey/IEEE2018_SNM/blob/master/figures/dist.pdf) are generated from this notebook.
In addition, Table I was generated from the biogeme code in the folder [`code/biogeme`](https://github.com/glederrey/IEEE2018_SNM/tree/master/code/biogeme).

## Authorization and Reference
> *We authorize the use of any part of the code at the condition that our article is explicitly mentioned using the following reference:* **Gael Lederrey, Virginie Lurking and Michel Bierlaire (2018). SNM: Stochastic Newton Method for Optimization of Discrete Choice Models. In *Proceedings of IEEE ITSC 2018.***
