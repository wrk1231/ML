1.ML paper review

(a)

(i). The inputs are radiomic features (gadolinium T1WI, T2WI, and FLAIR). The data is from braintumor MR imaging performed 9 months (orlater) post-radiochemotherapy performed by 2 institutions.

(ii). They are trying to distinguish radiation necrosis from recurrent brain tumors.

(iii). 2 classes. One is radiation necrosis and the other is recurrence necrosis.

(iv). 58. 43 for training and 15 for testing.

(v). They compared the outcomes of the SVM algorithms with the diagnosis of 2 professional neuroradiologists. The SVM is more accurate.

(b)

a) The inputs are 129450 clinical images and their labels(different kind of diseases). The data comes from different online respositories and Stanford University Medical Center, together with a pre-trained Google Architecture.

b) They try to distinguish different kinds of diseases based on the  images. 

c) 9.

d) 129450.  127463 for training and 1942 for testing.

e) They evaluated the performance by comparing the accuracy and variation of the model's performance with dermatologists' results. For the biopsy-proven part, they evaluated performance by testing  the accuracy, sensitivity, and compared it with the performance of the dematologists’.

2.Scalar Data Types
- Categorical: c), d), h) 
- Ordinal: b), g) 
- Interval: e)
- Ratio: a), f), i)

3.Vector representation of Binary variables
- $tr(Z) = \sum_{i}z_i^X$
- $\frac{1}{N}tr(Z) = \frac{\sum_{i}z_i^X}{N}$
- $Z^{XY}=Z^X*Z^Y$ represents the vector of the samples which both statement X and Y are true.
- $Z^X \cdot Z^Y$ represents the sum of cases in the samples when both statement X and Y are true.
- $\frac{1}{N}(tr(Z^X) - tr(Z^{XY}))$
- $tr(Z^X)+tr(Z^Y)-tr(Z^X * Z^Y)$
- $tr(Z^X)+tr(Z^Y)-2tr(Z^X * Z^Y)$

4.Matrix and Index Notation:

(a)$Y = X\Theta +\mathcal{E}$

(b) $\frac{1}{N}(Y - X\Theta)^T(Y-X\Theta)$

(c) $\frac{1}{N}\sum_{i=1}^{i=N}(y_i - \sum_{d=1}^{d=D}x_{i,d}\theta_d)^2$

(d) $\frac{\partial{E}}{\partial{\theta_d}} =0 \Leftrightarrow -2\sum_{i=1}^{i=N}(y_i - \sum_{d=1}^{d=D}x_{i,d}\theta_d)x_{i,d}=0$

(e) $\frac{\partial{E}}{\partial{\Theta}} = 0 \Leftrightarrow\frac{\partial{(Y - X\Theta)^T(Y-X\Theta)}}{\partial{\Theta}} \Leftrightarrow X^TX\Theta = X^TY$

5.Matrix and Index Notation II:

(a) $Y = XW^T +\mathcal{E}$

(b) $\frac{1}{NK}tr((Y - XW^T)^T(Y-XW^T))$

(c) $\frac{1}{NK}\sum_{k=1}^{k=K}\sum_{i=1}^{i=N}(y_{i,k} - \sum_{d=1}^{d=D}x_{i,d}w_{k,d})^2$

(d) $\frac{\partial{E}}{\partial{w_{k,d}}} = 0 \Leftrightarrow-2\sum_{i=1}^{i=N}(y_{i,k} - \sum_{d=1}^{d=D}x_{i,d}w_{k,d})x_{i,d} = 0$

(e)$\frac{\partial{E}}{\partial{W}}=0 \Leftrightarrow X^TXW^T = X^TY$
