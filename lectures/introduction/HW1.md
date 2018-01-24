3. Vector representation of Binary variables
* $tr(Z) = \sum_{i}z_i^X$
* $\frac{1}{N}tr(Z) = \frac{\sum_{i}z_i^X}{N}$
* $Z^{XY}=Z^X*Z^Y$ represents the vector of the samples which both statement X and Y are true.
* $Z^X \cdot Z^Y$ represents the sum of cases in the samples when both statement X and Y are true.
* $\frac{1}{N}(tr(Z^X) - tr(Z^{XY}))$
* $tr(Z^X)+tr(Z^Y)-tr(Z^X * Z^Y)$
* $tr(Z^X)+tr(Z^Y)-2tr(Z^X * Z^Y)$

4.Matrix and Index Notation:
* $Y = X\Theta +\mathcal{E}$
* $\frac{1}{N}(Y - X\Theta)^T(Y-X\Theta)$
* $\sum_{i=1}^{i=N}(y_i - \sum_{d=1}^{d=D}x_{i,d}\theta_d)^2$
* $\frac{\partial{E}}{\partial{\theta_d}} =0 \Leftrightarrow -2\sum_{i=1}^{i=N}(y_i - \sum_{d=1}^{d=D}x_{i,d}\theta_d)x_{i,d}=0$
* $\$