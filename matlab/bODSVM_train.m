function model = bODSVM_train(Y, X, param)

MAX_ITER = 12;
EPSILON = 1e-8;

% hyper-parameters and dataset info
[n, m] = size(X);
d = param.d;       % dimension
c = param.c;       % SVM penalty
l = param.lambda;   % alpha
g = param.gamma;  % gamma
G = param.G;       % graph matrix
svc_param = sprintf('-s 3 -c %.6f -q', c);

if m < d
    error("JDSVM:dimension", "Invalid input: m<d");
end

% initialize the projection matrix
rng(151);
P = randn(m, d);
Q = orth(P);

DG = sum(G);
Sa = full(X' .* DG * X);
Sb = full(X' * G * X);
Mi = l*Sa + g*eye(m);

for ii = 1:MAX_ITER
    %%% w step %%%
    % obtain embedding Z.
    XP = X*P;
    
    % train linear SVC in subspace
    svm = liblineartrain(Y, sparse(XP), svc_param);
    w = svm.w';
    
    %%% P step %%%
    options = optimoptions('quadprog','Display','none');
    Z = Y.*X;
    ZM = Z/Mi;
    w2 = sum(w.^2);
    H = w2 * ZM*Z';
    u = quadprog(0.5*(H+H'), l*ZM*Sb*Q*w-1, [], [], [], [], zeros(n, 1), c*ones(n, 1), [], options);
    P = Mi \ (l*Sb*Q + Z'*u*w');
    
    %%% Q step %%%
    [U, ~, V] = svd(Sb'*P, 'econ');
    Q = U*V';
end

model.P = P;
model.Q = Q;
model.svm = svm;

end