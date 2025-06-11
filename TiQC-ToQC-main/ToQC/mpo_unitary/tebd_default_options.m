function tebd_options=tebd_default_options()
tebd_options=struct('sv_min',10^(-10),'bond_dim',20,'bond_comp',20,'num_sweep',2,'num_midstep',1,'num_refined_step',1,'is_compressed',1,'is_second_order',0);
end