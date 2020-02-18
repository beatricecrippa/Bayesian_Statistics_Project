data
{
	int N; //observations
	vector[N] Y; //output
	real<lower=0> sigma; //variance of the residuals
	vector[3] mub;
	matrix[3,3] B;
	vector[2] G;
        real mu0;
}

parameters {
	vector[3] beta;
}

model
{	
	real csi;
	real alpha;
        matrix[N,N] V;
        vector[N] mu;

	beta ~ multi_normal(mub, B);
	csi = exp(beta[1]+beta[2]*G[1]+beta[3]*G[2])/(1+exp(beta[1]+beta[2]*G[1]+beta[3]*G[2]));
	alpha = (exp(csi)-1)/(exp(csi)+1);
        

        for(i in 1:N)
        {
          for(j in i:N)
          {
             if(j == i)
                {V[i,j] = sigma;}
             else
                {V[i,j] = alpha^(j+1-i);}
          }
          for (j in 1:(i-1))
              {V[i,j] = V[j,i];}
          mu[i] = mu0*alpha^(i-1);
        }

        Y ~ multi_normal(mu,V);
}
