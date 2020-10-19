# Bonferroni mean based fuzzy k-nearest neighbor classifier (BM-FKNN)
**Introduction:** <br/>
BM-FKNN is a new generalized version of the fuzzy k-nearest neighbor (FKNN) classifier that uses local mean vectors and utilizes the Bonferroni mean. 
The BM-FKNN classifier can be easily fitted for various contexts and applications, because the parametric Bonferroni mean allows for problem-based parameter
value fitting. The BM-FKNN classifier can perform well also in situations where clear imbalances in class distributions of data are found. 

**Matlab functions:** <br/>
The functions of the BM-FKNN algorithm (`BM_FKNN.m`), Bonferroni mean computation (`Bonferrni_mean`) are included. In addition to those files, 
an example (`Example.m`) of the use of BM_FKNN classifier is also presented. `Bonferroni_mean.m` is needed to compute Bonferroni mean vectors of the 
set of nearest neighbor in each class.<br/>

Reference:
    [Kumbure, M.M., Luukka,P.& Collan M.(2020) A new fuzzy k-nearest neighbor classifier based on
    the Bonferroni mean. *Pattern Recognition Letters*, 140, 172-178.](https://doi.org/10.1016/j.patrec.2020.10.005)<br/>
<br/>
 Created by Mahinda Mailagaha Kumbure & Pasi Luukka, 10/2020 <br/>
Based on Keller's definition of the fuzzy k-nearest neighbor algorithm. <br/>
