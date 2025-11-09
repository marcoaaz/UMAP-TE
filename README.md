# **UMAP-TE (extension)**

The Uniform Manifold Approximation and Projection - Trace elements (UMAP-TE) software tool can be used to study zircon chemical analysis. This extension is a functional programming version of version 1 (see [UMAP-TE](https://github.com/eblaup/UMAP-TE)). The repository can be used as a software tool to study the geochemistry of any mineral tabulated data and reduce the dimensionality with UMAP to better understand the intrinsic relationships within a big dataset. Note that any input data will be more useful if it has been curated and well-characterised (manually classified, filtered for outliers).

UMAP-TE extension allows to configure a mini-pipeline to produce interactive UMAP plots that increase the geological assessment reliability:

<p float="left" align="middle">
  <img src="https://github.com/user-attachments/assets/14d0e597-6b58-432f-935d-43e5daaaf107" width=75% height=75%>
  <img src="https://github.com/user-attachments/assets/797d75a5-5a4c-474c-9cbb-dd8b90b3ac27" width=75% height=75%>
</p>

The Python Jupyter notebooks partly reproduce the figures in the m/s: "Evaluation of non-linear dimensionality reduction for large geochemical datasets relevant to magmatic ore-fertility, petrologic classification, and provenance" (submitted to Chemical Geology). Please, cite this paper if trying to use any version of UMAP-TE. 
 
## Contents

We use existing literature dataset compilations of zircon geochemistry from:
* "Zircon_Fertility_Data-main/CG2024data_v5.csv"
  * Carrasco-Godoy et al., 2024 (https://doi.org/10.5382/econgeo.5086);   
* "Zircon_Fertility_Data-main/Test_Nathwani2024.csv"
  * Nathwani et al., 2024 (https://doi.org/10.1038/s41467-024-52786-5);

[UMAP-TE version 1](https://github.com/eblaup/UMAP-TE) (not available here): 
* "V3_zirconfertility" contains the script, data, and UMAP model used to project zircon geochemistry using Eu/Eu*, λ3, P, Dy/Yb, λ2, Ce/Nd, Eu, Tb, Gd, and Gd/Yb.
* "v3_zircongeochemistry" contains the script, data, and UMAP model used to project zircon geochemistry using P, Ce, Eu, Th, La, Pr, Y, Nd, Gd, Er, Yb, Sm, Dy, and Lu.

UMAP-TE extension (forked from version 1, 3-Nov-25):
* "loadingUMAP_and_plotting_v1.ipynb" can approximate UMAP-TE (version 1) results from "UMAP-TE_version1/UMAP_zircongeochemistry_PCA.ipynb" and "UMAP_zirconfertility_testing_PCA.ipynb"
* Does not process version 1 "Apatite" and "Yerington folders"
* Mini-pipeline to produce interactive UMAP plots using Dash.
  
## Instructions

To reproduce the results presented in the study, the user requires:
 - downloading the full repository.
 - installing a Python v3.9 environment. The 'requirements.txt' file is provided for installing the environment.
 - updating the script directory/file paths.
 - running using copy-pasted process metadata provided in "process_metadata.txt" within "loadingUMAP_and_plotting_v1.ipynb".
 - for reproducibility, it is recommended to save the script "process metadata" inputs in the same template for reutilisation.

The section 'Geochemical calculations' (in "loadingUMAP_and_plotting_v1.ipynb" and "UMAP_grid_search_v3.ipynb") requires using [pyrolite](https://pyrolite.readthedocs.io/en/main/examples/geochem/lambdas.html) and [Impute_REE](https://github.com/cicarrascog/imputeREE) (automatically calling R from CMD/Terminal). In addition, all process intermediate steps (standardisation scaler, UMAP transform, supervised Machine Learning model for binary classification) are saved in a determined 'trial' folder to avoid overwritting the outputs with different parameters. Note, using exactly the same order/columns names as in input table "Zircon_Fertility_Data-main/CG2024data_v5.csv" will reduce your workload in adapting your data to our code. For guidance, all scripts/functions were extensively commented.

### Interactive UMAP

The interactive UMAP mini-pipeline requires running:
1. "merge_Carrasco_db_v2.m" to fuse the original "Zircon Fertility Data.csv" and "External Validation.csv" tables ([Carrasco-Godoy et al., 2024](https://doi.org/10.5382/econgeo.5086)).
2. "UMAP_grid_search_v3.ipynb" to run UMAP multiple times on your dataset.
3. "interactive_UMAP_v1.ipynb" to generate the interactive UMAP plot in your web-browser. 

## Future updates

I could provide future custom functionality to this repository (see 'main_functions.py') based on user feedback. Contact maaz.geologia@gmail.com for installation issues, bugs and/or pull requests.

A cool next step would be implementing plots for seeing multi-class supervised learning boundaries (not only binary as below barren/fertile) on the 3D UMAP or PCA space:

<p float="left" align="middle">
  <img src="https://github.com/user-attachments/assets/9aed2326-497e-48c8-a21c-e1edb46af739" width=35% height=35%>
  <img src="https://github.com/user-attachments/assets/ca838fca-fe02-4bf7-b653-f69f0806b53e" width=35% height=35%>
</p>

Thanks  
Marco
