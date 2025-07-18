function [iF, iG] = infid_2q_propagator(H2q, H1q, x, mpo0, mpotg, varT, tebd_options)

% mpo0 should be a set of initial invariants, mpotg should be a set of targets
% x should be a set of coefficients [c11(t1), c11(t2), ... c11(tn), c12(t1), ...
% c1nc1(tn), c21(t1), ... c2nc2(tn), T]

% Calculates the infidelity and gradients for 1- and 2-qubit interaction
% control Hamiltonian given mpo0 = {I1, I2, ..., INc} and mpotg = {IT1,
% IT2, ... ITNc}

sv_min = tebd_options.sv_min;
D = tebd_options.bond_dim;
Dc = tebd_options.bond_comp;
nsweep = tebd_options.num_midstep;
nt = tebd_options.num_refined_step;
iscpr = tebd_options.is_compressed;
iso = tebd_options.is_second_order;

if iso == 1
    error('Function not yet set up for second order trotter!')
end

if varT == 1
    error('Function not yet set up for time duration optimisation')
end

d = 2; %automatically set to qubits!
Nc = length(mpo0); %number of initial/target invariants
n = length(mpo0{1});
nc1 = length(H1q);
nc2 = n - 1;


c = x(1: end - 1);
T = x(end);
nbin = length(c) / (nc1 + nc2);

c = reshape(c, [nbin, nc1 + nc2]);
c1 = c(:, 1:nc1);
c2 = c(:, (nc1 + 1):end);


dt = T / (nbin * nt);
Dt = T / nbin;


mpofw = cell(Nc, nbin + 1, n);
mpobw = cell(Nc, nbin + 1, n);
iG = zeros(nbin, nc1 + nc2);

for j = 1:Nc
    mpofw(j, 1, :) = mpo0(j);
    mpobw(j, 1, :) = mpotg(j);
end


for k = 1:nbin %gate construction at _control_ bin step
    %defining the forward and backwards gates
    %2 qubit forwards gate
    g2 = cell(1, n - 1);
    for j = 1:(n - 1)
        h = c2(k, j) * H2q{j};
        gate = expm(-1i * dt * h);
        gate = reshape(gate, [d,d,d,d]);
        g2{j} = gate;
    end
    %1 qubit forwards gate
    g1 = cell(1, n);
    for j = 1:n %chooses qubit
        h = zeros(d); %initially zero 2x2 matrix 
        for jc = 1:nc1
            for js = 1:length(H1q(jc).sys)
                if H1q(jc).sys(js) == j
                    h = h + c1(k, jc) * H1q(jc).op{js};
                end
            end
        end
        gate = expm(-1i * dt * h);
        g1{j} = gate;
    end

    for inv = 1:Nc %chooses invariant
        %forward propagation
        mpofw(inv, k + 1, :) = mpofw(inv, k, :); %HAVE RESIDUAL SINGLETON DIMENSION HERE WHICH BREAKS THE GATE_2Q_LR AS UNEXPECTED DIMENSION SIZES
        disp(size(mpofw))
        disp(size(mpofw(inv, :, :)))
        disp(size(mpofw(inv, 2, :)))
        disp(size(mpofw(inv, 2, 1)))
        %problem; size(mpofw(inv, k + 1, j)) = 1 1 4 != 1 4 
        %leads to issues with two qubit gate function

        %potential soluation; use at temporary MPO with the invariant
        %dimension squeezed away??? From first glance, this doesn't resolve
        %the issue
        %Perhaps depends on having a fourth index available? But playing
        %with the infid_2q function we didn't have this issue and that
        %seemed to have structre size(mpofw{k + 1, :}) = 1  4 
        
        mpofw_temp = squeeze(mpofw(inv, :, :));
        disp(size(mpofw_temp))
        disp(size(mpofw_temp(3, :)))
        disp(size(mpofw_temp(3, 1)))
        
        for jt = 1:nt
            %apply odd 2q terms
            for j = 1:2:n - 1
                
                [mpofw_temp{k + 1, j}, mpofw_temp{k + 1, j + 1}] = ...
                    gate_2q_LR(mpofw_temp{k + 1, j}, mpofw_temp{k + 1, j + 1}, g2{j}, sv_min, D);
            end
            %apply even 2q terms
            for j = 2:2:n - 1
                [mpofw{inv, k + 1, j}, mpofw{inv, k + 1, j + 1}] = ...
                    gate_2q_LR(mpofw{inv, k + 1, j}, mpofw{inv, k + 1, j + 1}, g2{j}, sv_min, D);
            end
            %apply 1q terms
            for j = 1:n
                [mpofw{inv, k + 1, j}] = gate_1q_LR(mpofw{inv, k + 1, j}, g1{j});
            end
        end
        %end of forwards propagation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %2 qubit backwards gate
        g2 = cell(1, n - 1); %used to save as g2bw, now dropped bw to overwrite g2 for space. hopefully this works
        for j = 1:n - 1      %the analogous thing was done for the 1q gates.
            h = c2(nbin - k + 1, j) * H2q{j};
            gate = expm(1i * dt * h);
            gate = reshape(gate, [d,d,d,d]);
            g2bw{j} = gate;
        end
        %1 qubit backwards gate
        g1 = cell(1, n);
        for j = 1:n
            h = zeros(d);
            for jc = 1:nc1
                for js = 1:length(H1q(jc).sys)
                    if H1q(jc).sys(js) == j
                        h = h + c1(nbin - k + 1, jc) * H1q(jc).op{js};
                    end
                end
            end
            gate = expm(1i * dt * h);
            g1{j} = gate;
        end
        
        mpobw(inv, k + 1, :) = mpobw(inv, k, :);
        %backwards propagation
        
        %For some reason with different nt dependence
        %CHANGED TO BE LIKE FORWARD PROPAGATION?!?! DOES THIS WORK!?!
        %this seeeeems to imply that either I've misunderstood something,
        %or the existing function is wrong?????
        for jt = 1:nt
            %apply 1 qubit terms
            for j = 1:n
                [mpobw{inv, k + 1, j}] = gate_1q_LR(mpobw{inv, k + 1, j}, g1{j});
            end
            %apply even 2 qubit terms
            for j = 2:2:n - 1
                [mpobw{inv, k + 1, j}, mpobw{inv, k + 1, j + 1}] = ...
                    gate_2q_LR(mpobw{inv, k + 1, j}, mpobw{inv, k + 1, j + 1}, gw{j}, sv_min, D);
            end
            %apply odd 2 qubit terms
            for j = 1:2:n - 1
                [mpobw{inv, k + 1, j}, mpobw{inv, k + 1, j + 1}] = ...
                    gate_2q_LR(mpobw{inv, k + 1, j}, mpobw{inv, k + 1, j + 1}, gw{j}, sv_min, D);
            end
        end
        if mod(k - 1, midstep) == 0 || k == nbin
            if iscpr == 1
                mpofw(inv, k + 1, :) = mpo_compress(mpofw(inv, k + 1, :), sv_min, Dc, nsweep);
                mpobw(inv, k + 1, :) = mpo_compress(mpobw(inv, k + 1, :), sv_min, Dc, nsweep);
            end
            mpofw(inv, k + 1, :) = mpo_normalize(mpofw(inv, k + 1, :));
            mpobw(inv, k + 1, :) = mpo_normalize(mpobw(inv, k + 1, :));
        end
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%finding the gradient
for inv = 1:Nc
    for k = 1:nbin
        tnsfw_temp = mpofw(inv, k + 1, :);
        tnsbw_temp = mpobw(inv, nbin - k + 1, :);
        
        %1 qubit control perturbation
        for jc = 1:nc1
            ovl_diff_left = 0
            for js = 1:length(H1q(jc).sys)
                jq = H1q(jc).sys(js);
                gate = -1i * Dt * H1q(jc).op{js};
                tnsfw_diff_left = tnsfw_temp;
                tnsfw_diff_left{jq} = gate_1q(tnsfw_diff_left{jq}, gate);
                ovl_diff_left = ovl_diff_left + mpo_overlap(tnsbw_temp, tnsfw_diff_left);
            end
            iG(k, jc) = iG(k, jc) -2 * real(ovl_diff_left) / Nc; %adds all inv derivatives together for a given control Hamiltonian
        end

        for jc2 = 1:nc2
            gate = -1i * Dt * H2q{jc2};
            gate = reshape(gate, [d,d,d,d]);

            tnsfw_diff_left = tnsfw_temp;

            [tnsfw_diff_left{jc2}, tnsfw_diff_left{jc2 + 1}] = ...
                gate_2q(tnsfw_diff_left{jc2}, tnsfw_diff_left{jc2 + 1}, gate, sv_min, D);

            ovl_diff_left = mpo_overlap(tnsbw_temp, tnsfw_diff_left);

            iG(k, nc1 + jc2) = iG(k, nc1 + jc2) - 2 * real(ovl_diff_left) / Nc;
        end
    end
end

iGT = 0; %not yet set up for optimising time scale evolution
iG = [iG(:); iGT];

ovl = 0;
for inv = 1:Nc
    ovl = ovl + real(mpo_overlap(mpotg{inv}, mpofw{inv, nbin + 1, :}));
end
iF = 1 - ovl;

end