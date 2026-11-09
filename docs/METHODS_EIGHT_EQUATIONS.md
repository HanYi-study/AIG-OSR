## 2. Methods

### 2.1. Observation-constrained underwater object reconstruction

We combine side-scan sonar (SSS) imagery and multibeam echo sounder (MBES) soundings to reconstruct the exposed surfaces of underwater objects and the surrounding seabed. SSS resolves acoustic texture and candidate structural boundaries on a dense image grid, whereas MBES supplies elevations in survey coordinates. Because acoustic intensity alone does not determine height and the soundings leave gaps around complex structures, we first infer a continuous elevation field between MBES controls. We then project this field onto the observed elevations while weakening the propagation of corrections across SSS-derived boundaries. Finally, we restrict the reported map to the neighbourhood of the soundings (Fig. 1).

![Observation-constrained reconstruction workflow](manuscript_assets/fig1_observation_constrained_reconstruction.png)

**Fig. 1 | Observation-constrained elevation reconstruction.** SSS supplies conditioning features and a candidate boundary mask; MBES supplies an elevation baseline and fixed controls. The conditioned neural field yields an initial raster, which is locally filtered to form \(\tilde Z\). We project this reference field onto the MBES controls and report the result within a raster support mask.

We represent the reconstructed surface as a height field, \(Z=f(x,y)\), where \(x\) and \(y\) are horizontal coordinates and \(Z\) is elevation. Each cell of the resulting digital elevation model (DEM) therefore has one elevation. We use additional points generated for rendering or meshing only as a visualization product; the height field does not recover multiple depths at one horizontal position or resolve vertical faces independently.

### 2.2. Acoustic observations and spatial alignment

We evaluate artificial reefs in Changyi, Shandong, and bridge piers in Ya’an, Sichuan, using paired SSS images and MBES point clouds (Fig. 2). The survey description identifies an SS4590 side-scan sonar and an MS8240 multibeam system carried by a hybrid aquatic–aerial platform. We use separately prepared MBES files labelled 100%, 50%, 10% and 5% retention. These are nominal **point-count** levels, not percentages of the seabed area lacking observations. After preprocessing and rasterization, the retained soundings define our elevation-control cells.

The survey imagery in manuscript Fig. 2 is omitted from this public code package. The 50%, 10% and 5% labels refer to point counts relative to the available MBES cloud.

We use the elevations stored in each MBES file. Before rasterization, we reject elevations more than ten standard deviations from the point-cloud mean and points whose mean three-dimensional distance to 20 nearest neighbours exceeds the dataset mean by two standard deviations. We aggregate soundings within each grid cell into one control elevation, taking the maximum by default. The archived metric DEMs use a grid spacing of 0.02 m. To obtain the baseline \(Z_{base}\), we interpolate the occupied cells linearly and fill locations outside their interpolation domain by nearest neighbour. We repeat the interpolation after excluding cells inside the SSS-derived object mask to obtain a seabed field. We filter that field with a 15 × 15 Gaussian kernel and the full-field interpolation with a 3 × 3 kernel, blend the two according to the object mask, and apply a final 3 × 3 Gaussian filter. When fewer than six seabed cells remain, we replace the seabed field by the minimum occupied control elevation. We retain the input MBES vertical reference without datum conversion.

We clip and log-transform the SSS intensity, subtract the global and column medians, and scale the result to \([0,1]\). From Scharr derivatives, intensity thresholds and connected components, we derive image gradients, a candidate object mask and a shadow mask. At horizontal location \(\mathbf p=(x,y)\), we assemble a six-channel descriptor \(\mathbf c(\mathbf p)\): normalized intensity; standardized horizontal and vertical gradients; standardized gradient magnitude; an illumination weight; and the sampled object-mask value. We use these image features to condition the elevation field and locate candidate graph boundaries, without treating intensity as an elevation observation.

We map the MBES bounding box linearly onto the SSS image extent and bilinearly sample the image features at mapped coordinates. This mapping assumes that the two inputs already cover the same axis-aligned footprint. We do not estimate residual rotation, image warping or slant-range corrections within the reconstruction.

### 2.3. SSS-conditioned continuous elevation field

We estimate the elevation left unexplained by \(Z_{base}\) with a three-layer periodic neural field. We condition its hidden layers on the SSS descriptor \(\mathbf c(\mathbf p)\) through feature-wise affine conditioning. For each horizontal location \(\mathbf p\), the network predicts a dimensionless normalized residual \(\Delta Z(\mathbf p)\). We convert it to elevation units before adding the baseline:

$$
Z_{pred}(\mathbf p)=Z_{base}(\mathbf p)
+\mu_{\Delta Z}+\sigma_{\Delta Z}\Delta Z(\mathbf p).
\tag{1}
$$

We denote the initial elevation estimate by \(Z_{pred}(\mathbf p)\) and the interpolated baseline at the same location by \(Z_{base}(\mathbf p)\). The quantities \(\mu_{\Delta Z}\) and \(\sigma_{\Delta Z}\) are the mean and standard deviation of \(Z_{ctrl}-Z_{base}\) over occupied MBES cells, with \(Z_{ctrl}\) denoting the rasterized sounding elevation. We feed normalized horizontal coordinates and the six SSS features into the conditioned network:

$$
\begin{aligned}
\Theta&=W_{proj}\operatorname{ReLU}(W_{ext}\mathbf c+\mathbf b_{ext})+\mathbf b_{proj},\\
(\boldsymbol\gamma_i,\boldsymbol\beta_i)&=\operatorname{reshape}_i(\Theta),\\
\mathbf h_i&=
\bigl[1+\alpha\tanh(\boldsymbol\gamma_i)\bigr]
\odot\sin\!\bigl[\omega_0(W_i\mathbf h_{i-1}+\mathbf b_i)\bigr]
+\alpha\tanh(\boldsymbol\beta_i),\\
\Delta Z(\mathbf p)&=W_{out}\mathbf h_{2}+\mathbf b_{out},
\qquad i=0,1,2.
\end{aligned}
\tag{2}
$$

We write the normalized horizontal coordinate as \(\mathbf h_{-1}\in[-1,1]^2\) and the 64-channel output of periodic layer \(i\) as \(\mathbf h_i\), for \(i=0,1,2\). We map the six-component descriptor \(\mathbf c\) to 64 channels with weights \(W_{ext}\) and bias \(\mathbf b_{ext}\), apply a rectified linear unit (ReLU), and project to the 384-component vector \(\Theta\) with \(W_{proj}\) and \(\mathbf b_{proj}\). We split \(\Theta\) into three pairs \((\boldsymbol\gamma_i,\boldsymbol\beta_i)\) of 64-component scale and shift vectors; \(\operatorname{reshape}_i\) denotes that selection. Within periodic layer \(i\), \(W_i\) and \(\mathbf b_i\) are the linear weights and bias, \(\omega_0=30\) sets the periodic frequency, and \(\alpha=0.1\) bounds the conditioning through \(\tanh\). The symbol \(\odot\) denotes component-wise multiplication. The final weights \(W_{out}\) and bias \(\mathbf b_{out}\) map \(\mathbf h_2\) to the scalar residual. This construction yields the initial field used by the metric projection.

We fit the field to MBES controls while discouraging large residuals and rapid variation away from candidate objects. The three effective training terms are

$$
\begin{aligned}
\mathcal L_{total}&=20\mathcal L_{data}+0.03\mathcal L_{base}
+0.003\mathcal L_{smooth},\\
\mathcal L_{data}&=\frac{1}{|\Omega_{gt}^{(t)}|}
\sum_{\mathbf p\in\Omega_{gt}^{(t)}}
\left[\Delta Z(\mathbf p)-
\frac{Z_{ctrl}(\mathbf p)-Z_{base}(\mathbf p)-\mu_{\Delta Z}}
{\sigma_{\Delta Z}}\right]^2,\\
\mathcal L_{base}&=\frac{1}{|\Omega_r^{(t)}|}
\sum_{\mathbf p\in\Omega_r^{(t)}}
\bigl[1-0.95\mathbb I_{mask}(\mathbf p)\bigr]\Delta Z(\mathbf p)^2,\\
\mathcal L_{smooth}&=\frac{1}{|\Omega_r^{(t)}|}
\sum_{\mathbf p\in\Omega_r^{(t)}}
\left\{\bigl[1-\mathbb I_{mask}(\mathbf p)\bigr]
\sqrt{\|\nabla_{\mathbf q}\Delta Z(\mathbf p)\|_2^2+10^{-8}}
+0.1\mathbb I_{mask}(\mathbf p)
\|\nabla_{\mathbf q}\Delta Z(\mathbf p)\|_2^2\right\}.
\end{aligned}
\tag{3}
$$

At optimization step \(t\), \(\Omega_{gt}^{(t)}\) is the MBES control-cell minibatch sampled with replacement, and \(\Omega_r^{(t)}\) is a separate batch of random grid locations; their cardinalities count sampled positions, including repetitions. We evaluate \(\mathcal L_{data}\) at the controls and \(\mathcal L_{base}\) and \(\mathcal L_{smooth}\) at the random locations. The sampled mask value \(\mathbb I_{mask}(\mathbf p)\) lies between zero and one after bilinear interpolation, so the baseline penalty is strongest outside candidate objects. In the smoothness term, \(\mathbf q\) denotes normalized horizontal coordinates and \(\nabla_{\mathbf q}\Delta Z\) is the network-output gradient with respect to those coordinates **while holding the sampled SSS descriptor fixed**; \(\|\cdot\|_2\) is the Euclidean norm. We optimize the weighted sum with Adam. Every tenth step we also compute an intensity–slope diagnostic and add it to the displayed scalar loss. Because its prediction is evaluated without gradient tracking, this diagnostic contributes no parameter gradient and does not enter the effective objective in Eq. (3).

### 2.4. Boundary-aware projection onto metric controls

The learned field predicts elevations at unsounded cells, but its values at MBES controls need not match the observations exactly. We first make a locally filtered reference field, gradually increasing the contribution of a 3 × 3 Gaussian filter with distance from the nearest occupied control cell:

$$
\begin{aligned}
W_{smooth}(\mathbf p)&=\operatorname{clip}
\left(\frac{d_{px}(\mathbf p,P_{MBES})}{5},0,1\right),\\
\tilde Z(\mathbf p)&=
\bigl[1-W_{smooth}(\mathbf p)\bigr]Z_{pred}(\mathbf p)
+W_{smooth}(\mathbf p)
\operatorname{GaussianBlur}_{3\times3}(Z_{pred})(\mathbf p).
\end{aligned}
\tag{4}
$$

We let \(P_{MBES}\) contain the occupied MBES raster cells and measure \(d_{px}(\mathbf p,P_{MBES})\), their approximate Euclidean raster distance from cell \(\mathbf p\), in pixels. Clipping this distance divided by five to \([0,1]\) gives the smoothing weight \(W_{smooth}\). We denote the local 3 × 3 Gaussian filter of \(Z_{pred}\) by \(\operatorname{GaussianBlur}_{3\times3}(Z_{pred})\), and combine it with \(Z_{pred}\) to form \(\tilde Z\), the reference for projection. At a control cell, \(W_{smooth}=0\); the filter leaves its network prediction unchanged before we impose the observed elevation.

We next connect each DEM cell to its eight immediate neighbours. We give links within the same object-mask class a weight of 1 along grid rows or columns and 0.5 along diagonals. For links crossing the candidate boundary, we reduce the weight to 0.001:

$$
W_{ij}=\begin{cases}
\omega_{spatial}(i,j),
&(i,j)\in\mathcal E,\ \mathbb I_{mask}(i)=\mathbb I_{mask}(j),\\
\epsilon_{cut},
&(i,j)\in\mathcal E,\ \mathbb I_{mask}(i)\ne\mathbb I_{mask}(j),\\
0,&(i,j)\notin\mathcal E.
\end{cases}
\tag{5}
$$

We index grid cells by \(i\) and \(j\), call the set of eight-neighbour links \(\mathcal E\), and write their weights as \(W_{ij}\), taking zero for unconnected cells. The within-class weight \(\omega_{spatial}(i,j)\) equals 1 for orthogonal neighbours and 0.5 for diagonal neighbours. We resize the SSS object mask to the DEM grid and threshold it at 0.5 to obtain the binary class \(\mathbb I_{mask}(i)\) used in this graph. The cross-class weight \(\epsilon_{cut}=0.001\) weakens these links, so an image boundary changes how corrections spread without asserting a physical elevation discontinuity.

We project \(\tilde Z\) onto the MBES elevations by penalizing graph-Laplacian variation in the correction, rather than smoothing the final elevation field itself. Let \(W\) be the matrix of the weights in Eq. (5), \(\mathbf1\) the all-ones vector, \(D=\operatorname{diag}(W\mathbf1)\) the weighted degree matrix, and \(L=D-W\) the graph Laplacian. We compute

$$
\begin{aligned}
Z^*&=\underset{Z}{\operatorname{arg\,min}}
\frac12\bigl\|L(Z-\tilde Z)\bigr\|_2^2,\\
\text{subject to}\quad Z_k&=Z_{ctrl}(k),
\qquad k\in\mathcal V_K.
\end{aligned}
\tag{6}
$$

We vectorize the candidate, reference and corrected rasters as \(Z\), \(\tilde Z\) and \(Z^*\). We minimize half the squared Euclidean norm \(\|L(Z-\tilde Z)\|_2^2\) while fixing every control cell \(k\) in \(\mathcal V_K\) to its MBES elevation \(Z_{ctrl}(k)\). Our implementation constructs \(W-D=-L\), whose squared action gives the same objective. The reference field thus enters the adjustment while the observations determine the anchored values. Writing \(U\) for free cells and \(K\) for control cells gives the sparse system

$$
(L^{\top}L)_{UU}Z_U
=\bigl(L^{\top}L\tilde Z\bigr)_U
-(L^{\top}L)_{UK}Z_K,
\qquad Z_K=Z_{ctrl}|_{\mathcal V_K}.
\tag{7}
$$

We use \(L^{\top}\) for the transpose of \(L\); \(L^{\top}L\) is the squared-Laplacian matrix. The subscripts \(UU\) and \(UK\) select its free–free and free–control blocks, and \((L^{\top}L\tilde Z)_U\) selects the free-cell entries of the reference-field term. We solve for the free elevations \(Z_U\) by sparse conjugate gradients by default and then assign the anchored elevations \(Z_K\) directly from the MBES controls. A unique free-cell solution requires at least one control in every connected graph component. We perform this projection after training the neural field; the cross-boundary links remain weak but nonzero.

### 2.5. Spatial support and map output

We limit the reported DEM to cells near an occupied MBES control. To do so, we compute a raster distance transform and retain the projected elevation where

$$
\Omega_{supp}=\{\mathbf p\in\Omega_{grid}:
\Phi(\mathbf p)\le\tau\},
\qquad
\tau=\max\!\left(2,
\left\lfloor\frac{D_{hole}}{Step}\right\rfloor\right).
\tag{8}
$$

We denote all DEM cells by \(\Omega_{grid}\) and the retained cells by \(\Omega_{supp}\). For each cell \(\mathbf p\), \(\Phi(\mathbf p)\) gives its approximate distance in **pixels** to the nearest occupied MBES cell. We keep it when \(\Phi(\mathbf p)\leq\tau\), where \(\tau\) is the pixel threshold obtained by dividing our chosen metric radius \(D_{hole}\) by grid spacing \(Step\), taking the floor \(\lfloor\cdot\rfloor\), and enforcing a minimum of two pixels. The reference settings are \(D_{hole}=0.30\) m and \(Step=0.02\) m. We mark all other cells invalid and use the retained footprint only to delimit the reported DEM, without treating either mask as a calibrated uncertainty estimate.

We export the retained elevations as XYZ points in the input point cloud's coordinate frame. Because the XYZ output carries no coordinate-reference or vertical-datum metadata, its spatial interpretation follows the accompanying survey information. The resulting map depends on the alignment of the SSS image and MBES soundings, the aggregation of control elevations and the extracted boundary mask. We therefore interpret AIG-OSR as a locally constrained reconstruction of single-valued underwater object surfaces.
