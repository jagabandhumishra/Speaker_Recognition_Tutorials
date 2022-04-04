function score = gpldaScore(gpldaModel,w1,wt)
% Center the data
w1 = w1 - gpldaModel.mu;
wt = wt - gpldaModel.mu;

% Whiten the data
w1 = gpldaModel.WhiteningMatrix*w1;
wt = gpldaModel.WhiteningMatrix*wt;

% Length-normalize the data
w1 = w1./vecnorm(w1);
wt = wt./vecnorm(wt);

% Score the similarity of the i-vectors based on the log-likelihood.
VVt = gpldaModel.EigenVoices * gpldaModel.EigenVoices';
SVVt = gpldaModel.Sigma + VVt;

term1 = pinv([SVVt VVt;VVt SVVt]);
term2 = pinv(SVVt);

w1wt = [w1;wt];
score = w1wt'*term1*w1wt - w1'*term2*w1 - wt'*term2*wt;
end