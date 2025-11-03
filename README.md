# **Update of UMAP-TE**

We provide the Uniform Manifold Approximation and Projection - Trace elements (UMAP-TE) software as a new tool to process zircon/apatite trace element chemistry but the approach is generalisable to any geochemical tabulated data of known origin (curated/annotated/filtered for outliers). This is an update of version 1 provided in the initial m/s submission that includes a mini-pipeline to produce interactive UMAP plots:

<p float="left" align="middle">
  <img src="https://github.com/user-attachments/assets/14d0e597-6b58-432f-935d-43e5daaaf107" width=75% height=75%>
  <img src="https://github.com/user-attachments/assets/797d75a5-5a4c-474c-9cbb-dd8b90b3ac27" width=75% height=75%>
</p>

This repository has the code and data tables used in the submitted m/s: "Evaluation of non-linear dimensionality reduction for large geochemical datasets relevant to magmatic ore-fertility, petrologic classification, and provenance" (Chemical Geology). 
 
## Contents

This repository uses existing literature dataset compilations of apatite and zircon geochemistry from:
* "Zircon_Fertility_Data-main/CG2024data_v5.csv"
  * Carrasco-Godoy et al., 2024 (https://doi.org/10.5382/econgeo.5086) ;   
* "Zircon_Fertility_Data-main/Test_Nathwani2024.csv"
  * Nathwani et al., 2024 (https://doi.org/10.1038/s41467-024-52786-5) ;
* "UMAP-TE_version1/Apatite/OSullivan-etal_2019.csv"
  * Castellanos-Melendez et al., 2024 (https://doi.org/10.1016/j.epsl.2024.119053) ;
* "UMAP-TE_version1/Yerington/trials/Yerington_test.csv"
  * O'Sullivan et al., 2020, downloaded from their PANGAEA repository (https://doi.org/10.1594/PANGAEA.906570).

UMAP-TE version 1:
* "UMAP-TE_version1/V3_zirconfertility" contains the script, data, and UMAP model used to project zircon geochemistry using Eu/Eu*, λ3, P, Dy/Yb, λ2, Ce/Nd, Eu, Tb, Gd, and Gd/Yb.
* "UMAP-TE_version1/v3_zircongeochemistry" contains the script, data, and UMAP model used to project zircon geochemistry using P, Ce, Eu, Th, La, Pr, Y, Nd, Gd, Er, Yb, Sm, Dy, and Lu.
* "UMAP-TE_version1/Yerington" contains the scripts and data used to construct the Yerington district figures.
* "UMAP-TE_version1/Apatite" contains the script and dataset used to project the apatite data of O'Sullivan et al., 2020. 

UMAP-TE version 2:
* "loadingUMAP_and_plotting_v1.ipynb" can reproduce version 1 results of "UMAP-TE_version1/UMAP_zircongeochemistry_PCA.ipynb" and "UMAP_zirconfertility_testing_PCA.ipynb"
* Mini-pipeline to produce interactive UMAP plots using Dash.
  
## Instructions

The jupyter notebook(s) and main_function.py files can be downloaded and run on the users computer. To reproduce the results presented in the study, the user requires downloading the full repository, installing a Python v3.9 environment, updating the script directory/file paths, and running using copy-pasted process metadata provided in "process_metadata.txt" within "loadingUMAP_and_plotting_v1.ipynb". 

The 'requirements.txt' file is provided for installing the environment. The new section 'Geochemical calculations' (in "loadingUMAP_and_plotting_v1.ipynb" and "UMAP_grid_search_v3.ipynb") requires using [pyrolite](https://pyrolite.readthedocs.io/en/main/) and [Impute_REE](https://cran.r-project.org/web/packages/imputeREE/index.html) (automatically calling R from CMD/Terminal).

Users are encouraged to transform their own datasets using the script. For reproducibility, all process intermediate steps (standardisation scaler, UMAP transform, supervised Machine Learning model for binary classification) will be saved in a determined 'trial' output folder to avoid overwritting the outputs with different parameters. Note, using exactly the same order/columns names as in input table "Zircon_Fertility_Data-main/CG2024data_v5.csv" will reduce your workload in adapting your data to our code. For guidance, all scripts/functions were extensively commented.

### Interactive UMAP

The mini-pipeline (and this repository) requires running:
1. "merge_Carrasco_db_v2.m" to fuse the original "Zircon Fertility Data.csv" and "External Validation.csv" tables.
2. "UMAP_grid_search_v3.ipynb" to run UMAP multiple times on your dataset.
3. "interactive_UMAP_v1.ipynb" to generate the interactive UMAP plot in the web-browser. 

## Future updates

Marco Acevedo and Elise Laupland will provide future functionality and updates to this repository based on user feedback. For immediate questions, please, contact Marco Acevedo Zamora (maaz.geologia@gmail.com)

The precise repository description will be updated as the m/s is still in review (e.g., Figures numbers and final versions). 

Cheers,
