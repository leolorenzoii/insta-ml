# Insta ML For MSDS

A consolidated repository of the different tools to implement predictive analytics using machine learning (from the _Fundamentals of Machine Learning for Predictive Data Analytics 2e_ Kelleher et. al. 2020) based on the CRoss-Industry Standard Process for Data Mining (CRISP-DM).

<img src="./images/crisp-dm.png" alt="Crisp DM Flowchart" width="550"/>

*This is repository has been prepared and conceptualized by LLorenzo with the help of PDonato.*

## Quick Start

1. Install an environment for the project

```
conda env create -f environment.yml
```

or

```
mamba env create -f environment.yml
```

2. Set the correct filepath of the data on the `config.yml` file

```
filepaths:
  data: '<filepath to dataset>'
```

3. Use the notebooks to guide you through the whole predictive analytics using machine learning pipeline.

## References

- Kelleher JD, Mac Namee B, D'arcy A. Fundamentals of machine learning for predictive data analytics: algorithms, worked examples, and case studies. MIT press; 2020 Oct 20.
- https://auto.gluon.ai/stable/index.html
- https://docs.profiling.ydata.ai/latest/
