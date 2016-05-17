require 'nngraph'
require 'KLDCriterion'
require 'Sampler'
require 'gnuplot'

x_dim = 10
y_dim = 10
hid_dim = 100
z_dim = 2

--prior network----------------------------------------
x = nn.Identity()()
p_hid = nn.ReLU(true)(nn.Linear(x_dim,hid_dim)(x))
p_mu = nn.Linear(hid_dim,z_dim)(p_hid)
p_log_sig = nn.Linear(hid_dim,z_dim)(p_hid)
prior_network = nn.gModule({x},{p_mu,p_log_sig})
--inference/recognition network (encoder)---------------
x = nn.Identity()()
y = nn.Identity()()
q_hid = nn.ReLU(true)(nn.Linear(x_dim+y_dim,hid_dim)(nn.JoinTable(2){x,y}))
q_mu = nn.Linear(hid_dim,z_dim)(q_hid)
q_log_sig = nn.Linear(hid_dim,z_dim)(q_hid)
inference_network = nn.gModule({x,y},{q_mu,q_log_sig})
--generative network (decoder)------------------------
x = nn.Identity()()
z = nn.Identity()()
g_hid = nn.ReLU(true)(nn.Linear(x_dim+z_dim,hid_dim)(nn.JoinTable(2){x,z}))
g_y = nn.Sigmoid(true)(nn.Linear(hid_dim,y_dim)(g_hid))
generative_network = nn.gModule({x,z},{g_y})
--training network-----------------------------------
--generate
x = nn.Identity()()
y = nn.Identity()()
q_z = inference_network{x,y}
z = nn.Sampler()(q_z)
g_y = generative_network{x,z}
--compare to prior
p_z = prior_network{x}
kl_cost = nn.KLDCriterion(){p_z,q_z}
network = nn.gModule({x,y},{g_y,kl_cost})
--sampling network-----------------------------------
x = nn.Identity()()
p_z = prior_network{x}
g_y = generative_network{x,p_z}
sampling_network = nn.gModule({x},{g_y})



w,dw = network:getParameters()
o = network:forward{torch.rand(13,x_dim),torch.rand(13,y_dim)}
network:backward({torch.rand(13,x_dim),torch.rand(13,y_dim)},{torch.zeros(13,y_dim),torch.zeros(1)})




