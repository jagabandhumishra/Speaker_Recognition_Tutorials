function[p]= gauss_prob(x,m,v)

[D d]=size(x);

p=(1/((2*pi)^(D/2)*(det(v))^.5))*exp(-.5*(x-m)'*inv(v)*(x-m));
