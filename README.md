# METAL ADDITIVE MANUFACTURING
# Local monitoring and defects prediction via Bayesian approach

The metal additive manufacturing techinque consists of the melting of layers of metal powder that, once solidified, form a desired 3D shape. We have monitored the printing process of a triangular shape using a non termic camera and estracted the intesity values of each pixel at every frame from the obtained gray scale video. We have isolated the triangular shape, localized the laser position (with criteria based on the intensity, area and non-statisticity of bright sposts) ans set to 0 the corresponding intensity.
In the folder Dati Iniziali you can find the video, the Python code used for the data etraction from the video, the dataset (Dataset preliminari Tr1.xlsx) and the coordinates of each pixel in the image (Coord.xlsx).

Folder "Processing":

In the initial dataset there are some 0 intensity values that do not correspond to the laser but to some sparkles, that cover the real intensity values: we have simulated these missing values via MCMC Gibbs sampling.
The variable describing the intensity of pixel p at frame f is denoted by Vpf, and we have supposed that for every p the intensities at each frame are independent and identically distributed as a gaussian of mean mup and variance sigmap^2; mup is apriori distributed as a Gaussian of mean mu0p and variance tau^2. mu0p, tau^2 and sigmap^2 are fixed. (see slide 5 of the presentation workshop3.pdf for the mathematical description)
Then we have performed a rotation on the coordinates so that we got the same image as in the video, and performed a dimensionality reduction on the dataset by considering only the top-right angle of the triangle (from 996 pixels to 420).

This simulation is in the file Processing.R and the full datasets are Dataset_definitivo_gibbs.xlsx and Coordinates.xlsx, while the reduced ones are Dataset_cut_gibbs.xlsx and Coord_cut.xlsx.

In the file Data_extraxtion.R we have extraced all the variables that we have actually used for our models.

We have considered the intervals between two consecutive passages of the laser on each pixel, and from each one we gain one observation for each of the following variables:
- mean intensity in the interval
- number of passages of the laser up to the beginning of the interval (the laser doesn't follow a specific path but jumps from point to point, so we can assume it to be random)
- number of frames with laser on up to the beginning of the interval (the laser may pass quickly or insist for many frames on the single pixel)
- cooling time at the previous interval (time between the maximum intensity value after the firt passage of laser and the first frame where the intensity value is below the global empirical intensity mean)
- geometrical covariate number 1 (geom) describes the mass of triangle that surrounds each pixel and in particular the amplitude of the angle where the each pixel belongs
- geometrical covariate number 2 (g) is the distance of each pixel from the farther vertex.
Here you can also find the script for many graphical representaions of the extracted varibles and how we have set the hyperparameters for the two models using frequentist estimates.

Folder "Model1":

Here we have saved all the variables extracted in Data_extraction.R and the hyperparameters for the first model as files .dat.


Goal: describe which variables influence the light intensity the most.

Bayesian gaussian hyerarchical model:

Z_p | β ,Q_p   ~ ind N_(n_p ) (x_p^t β,Q_p)

π(β,Q_p,δ_p )=π(β)π(Q_p |δ_p)π(δ_p)

β ~ iid N_4 (0,B)

Q_p | δ_p  ~ δ_p IW(η_0,Q_(0_p)^(-1) )+(1-δ_p ) σ_0^2 I_(n_p )

δ_p |ξ_p~Be(ξ_p)

We have divided all the intensity values into 420 groups, each one representing a pixel:
In each group there is a different number np of data, corresponding to the observations in the considered pixel.

In each group we describe the intensity Zp using a linear model, with covariates Np (number of frames with laser on), Lp (the number of passages of laser up to each considered time interval), Tp (latest cooling time): Z_p=x_p^t β+ ε_p, where the residuals ε_p are assumed to be Gaussian distributed with mean 0 and covariance Qp.

The file model1.R simulates this model from the file model1.STAN (2 chains, 2000 iterations, thin = 2).
STAN doesn’t support spikes in the parameters distributions, we have integrated out the delta and used directly the parameter of the Bernoulli in the expression of the mixture on Qp.

In the image beta.png we observe that the posterior of the regression parameters is actually gaussian and that the main dependecy is on the number of passages of the laser, that is actually negative: this can be interpreted as if a higher number of visits from the laser makes the final intensity lower. The dependence on the number of frames with laser on, that is the actual time spent by the laser insisting on a pixel, is very low, but positive: intuitively, if the laser spends much time heating a pixel its temperature is higher, and the risk of burns increases. Finally, while we expected an almost linear dependence on the cooling time, it is positive but yet very low.

The images traceplot.png and ac.png represent the traceplot of the regression parametrs and the autocorrelation plots. The traceplots of the regression parameters are very thick and that their autocorrecation is low and appear to coverge to 0, so we can assert that our simulation is good.

In conclusion, if we want to control the burning problems, a large number of passages of the laser isn’t a problem at all, provided that each visit lasts not too much. Moreover, the cooling time is just a little contribution to the next intensity value.


Floder "Model2":

Solidification problems in metal additive manufacturing may arise also if the next layer of powder Is set even if the previous one hasn’t solidified yet: indeed, if we add powder on a very hot base, this will start to melt in the corresponding spots, and another passage of the laser on it may cause burns. In this case the next intensity value would be very different from the previous one and it is unpredictable without further information, while if we are in a non-critical spot the intensity values wouldn’t vary too much from one interval to the other and the dependence on the previous observation would be almost linear.
We have decided to monitor the intensity values through an autoregressive model, with parameter in (-1,1).
Here we consider each pixel independent of the others and we want to describe the mean intensity in every interval with a linear dependence on the previous observation:
V_pt= α_p V_(p,t-1)+ε_pt,       ε_pt  ~iid N(0,σ^2)
Where the regression parameter is once again dependent on the geometrical features of our image: 
α_p=2/(1+e^(-ξ_p ) )-1=tanh⁡〖( ξ_p/2  )〗
This is a sigmoidal transformation of the logit parameter ξ_p , described in the previous model, so that its range is (-1,1).
We have decided to make use of the geometrical properties because we suppose that high intensities will lead to burning problems, and the critical points for burning are related to the position and described by this parameter.

Conditionally to the previous observed one, each intensity value is distributed as follows:

V_p1 |α_p,σ^2  ~ N(μ_0,σ^2)
V_pj | V_(p,j-1),α_p,σ^2  ~ N(α_p^(j-1) V_(p,j-1),σ^2 )        ∀ j=2,3,…,n_(p_tot )

The joint distribution we have simulated from is the following:

V_p |α_p,σ^2   ~ N_(p_tot ) (μ_p,V_p)

Where μ_p,i = α_p^(j-1) * μ_0 and Vp,i,i = σ^2, Vp,i,j = α_p^(j-1+i) ∀ j>i, Vp is symmetric.
σ^2 and μ_0 are fixed using their frequentist estimates (in file Data_extraction.R, folder "Processing").
Note that, since α_p∈(-1,1), the correlation between two different observations decreases with their distance in time.

The file model2.R implements the simulation from model2.STAN, with 10000 iteration and thinning 10, 2 chains, for every pixel.
We have monitored the logit parameters, so that we could understand how the geometry could affect the mean intensity.

In the files traceplot.png and ac.png you can observe that the traceplots are very thick and the autocorrelation stays in the range (-0.1, 0.1) and converges to 0.

We have focues on four pixels:
p <- 170   median value of geom and g
p <- 147  min value of geom and max value of g 
p <- 188  max value of geom
p <- 406  min value of g
and in the corresponding traceplot.png ac.png, beta.png and alpha.png are represented, respectively, the traceplots, the autocorrelation plots and the empirical posterior densities of the logit parameters, and the empirical density of the autocorrelation regression parameter α_p.

The most interesting result is about the parameter alpha, that describes the dependence among the observations. The probability of a negative value of alpha is null, and in particular for pixel 188 a large mass is put on values near 1, while if we consider the pixel 147 the mass is all concentrated at 0, and the probability of values greater than 0.05 is null. For pixel 406 high mass is put for values of α_p < 0.5, but not 0, while for pixel 170 there is high probabiliy for values grater than 0.5.

Interpretation:
-	147 – min geom and max g – far from the vertex, high mass around the pixel
 Alpha depends more on G1 (angle amplitude) and it is almost zero: no dependence on the previous observation

-	188 – max geom – high mass around the pixel
Alpha depends mostly on g (distance from vertex) and it is very high: significant dependence on the previous observation

-	170 – median geom and g
Effect similar to 147 (little geom) but moderate

-	406 – max g (max distance from a vertex)
Effect similar to 188 (high geom)  but moderate

Conclusion:
The least dependence on the previous observation is recovered when on the vertex, so in order to prevent burns we can assert that it is better to avoid acute angles and the boundary is critical, and in this case the negative dependence on g confirms that the points become less critical when far from the vertex.
