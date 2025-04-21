function [model, obj] = mODSVM_train(Ylabel, X, param, varargin)

% parsing
p = inputParser;
p.KeepUnmatched = true;
p.addParameter('qpsolver', 'csmo');
p.addParameter('maxiter', 10);
p.addParameter('tol', 1e-5);
p.addParameter('initP', 'pca');
p.addParameter('maxQP', 3);
parse(p, varargin{:});

MAX_ITER = p.Results.maxiter;
EPSILON = p.Results.tol;

% hyper-parameters and dataset info
[n, m] = size(X);
d = param.d;       % dimension
c = param.c;       % SVM penalty
l = param.lambda;  % alpha
g = param.gamma;   % gamma
G = param.G;       % graph matrix
svc_param = sprintf('-s 4 -c %.6f -q', c);
k = numel(unique(Ylabel));

if m < d
    error("JDSVM:dimension", "Invalid input: m<d");
end
% objective value list
obj = zeros(MAX_ITER, 1);
obj_QP = zeros(MAX_ITER, p.Results.maxQP);

% initialize the projection matrix
rng(151); % this is my lucky number
if isstring(convertCharsToStrings(p.Results.initP))
    switch lower(p.Results.initP)
        case 'random'
            P = randn(m, d);
            Q = orth(P);
        case 'pca'
            [U, ~, ~] = svd(X','econ');
            P = U(:, 1:d);
            Q = P;
    end
elseif ismatrix(p.Results.initP)
    P = p.Results.initP;
    if norm(P'*P - eye(d)) < 1e-6
        Q = P;
    else
        Q = orth(P);
    end
else
    error('I dont want to waste my time to write the error message.')
end

A = zeros(n, k);
d = size(P, 2);

% precompute the constants
DG = sum(G);
Sa = X' .* DG * X;
Sb = X' * G * X;
Mi = l*Sa + g*eye(m);
K = X / Mi * X';
Hleft = l * X / Mi * Sb;
Y = ind2vec(Ylabel')';
trXDX = trace(Sa);

qpoptions = optimoptions('quadprog', 'Display', 'none');

% init W
svm = liblineartrain(Ylabel, sparse(X*P), svc_param);
% [~, accu, ~] = liblinearpredict(Ylabel, sparse(X*P), svm, '-q');
% disp(accu);
W = svm.w';
Wmax = zeros(n, 1);

for lp = 1:MAX_ITER
    WPX = X*P*W;
    for ii = 1:n
        Wmax(ii, :) = max(WPX(Y(ii, :) == 0));
    end
    Wyi  = WPX(Y > 0);
    
    obj(lp) = 0.5*sum(W(:).^2) + ...
        c * sum(max(0, Wmax-Wyi+1)) + ...
        l*0.5*(sum((Sa*P).*P, 'all') - 2*sum((Sb*P).*Q, 'all') + trXDX) + ...
        g*0.5*sum(P(:).^2);

    if lp > 1
        if abs(obj(lp) - obj(lp-1)) / obj(lp) < EPSILON
            break
        end
    end

    %%% W step %%%
    if lp > 1
        XP = X*P;
        svm = liblineartrain(Ylabel, sparse(XP), svc_param);
        W = svm.w';
    end
    
    phi = W'*W;
    
    %%% P step %%%
    H = Hleft * Q*W - Y;

    % quadprog for large-scale QPP of A
    % Empirically T=3 can reach the minimum
    for lpQP = 1:p.Results.maxQP
        % recording the QP objective function
        obj_QP(lp, lpQP) = trace(A'*K*A*phi) + sum(A.*H, "all");

        % start BCD iteration
        for ii = randperm(n)
            if K(ii, ii) <= eps
                A(ii, :) = zeros(1, k);
                continue;
            end
            
            Ki = K(:, ii);
            Ki(ii) = 0;
            nu = H(ii, :)' + phi*A'*Ki;
            
            [~, max_idx] = max(nu);
            if max_idx == Ylabel(ii)
                % zero solution checking
                A(ii, :) = zeros(1, k);
            else
                switch p.Results.qpsolver
                    case 'quadprog'
                        A(ii, :) = quadprog(K(ii, ii) * phi, nu, eye(k), c*Y(ii, :)', ones(1, k), 0, [], [], A(ii, :), qpoptions);
                    case 'csmo'
                        A(ii, :) = Csubqp_smo(K(ii, ii) * phi, nu, c, int32(Ylabel(ii)));
                    case 'matlab-smo'
                        A(ii, :) = smo_subqp(K(ii, ii) * phi, Ylabel(ii), c, nu);
                end
            end
        end
    end

    P = Mi \ (l*Sb*Q + X'*A*W');
    
    %%% Q step %%%
    [U, ~, V] = svd(Sb'*P, 'econ');
    Q = U*V';
end

model.P = P;
model.Q = Q;
model.svm = svm;
model.obj = obj;
model.objQP = obj_QP;

end

function a = smo_subqp(Q, yl, c, tau)
MAX_SMO_ITER = 200;
k = size(Q, 1);
a = zeros(k, 1);
cy = sparse(yl, 1, c, k, 1);

r = yl;
% [~, s] = max(tau);
s = randi(k);

% main loop routine
for lp = 1:MAX_SMO_ITER
    lt = cy(r) - a(r);
    gt = a(s) - cy(s);

    % 
    q = Q(r,r) + Q(s,s) - 2*Q(r,s);
    p = (Q(r,:)-Q(s,:))*a + tau(r) - tau(s);
    % a(r)*Q(r,r) - a(s)*Q(s,s) + Q(r,s)*(a(r) - a(s)) + tau(r) - tau(s);

    if q < eps
        if p >= 0
            delta = gt;
        else
            delta = lt;
        end
    else
        delta = min(max(-p / q, gt), lt);
    end
    a(r) = a(r) + delta;
    a(s) = a(s) - delta;

    % check the optimal condition
    grad = Q*a + tau;
    [rho_max, idmx] = max(grad);
    inactive = find(a < cy);
    [rho_min, idmn] = min(grad(inactive));
    if rho_max <= rho_min + 1e-4
%         fprintf('.');
        return
    end
    
    % heuristic strategy 1
    r = inactive(randi(numel(inactive)));
    s = idmx;

    % heuristic strategy 2
%     r = inactive(idmn);
%     s = idmx;

%     %
%     r = randi(k);
%     s = randi(k);
end
fprintf(';');
end