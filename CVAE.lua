require 'nngraph'
require 'KLDCriterion'
require 'Sampler'
require 'gnuplot'
require 'optim'
time = sys.clock()

x_dim = 10
y_dim = 10+2
hid_dim = 10000
z_dim = 2
gen_layers = 1
prior_layers = 1
infer_layers = 1

--prior network----------------------------------------
x = nn.Identity()()
p_hid = nn.ReLU(true)(nn.Linear(x_dim,hid_dim)(x))
for i=1,prior_layers-1 do
    p_hid = nn.ReLU(true)(nn.Linear(hid_dim,hid_dim)(p_hid))
end
p_mu = nn.Linear(hid_dim,z_dim)(p_hid)
p_log_sig = nn.Linear(hid_dim,z_dim)(p_hid)
prior_network = nn.gModule({x},{p_mu,p_log_sig})
--inference/recognition network (encoder)---------------
x = nn.Identity()()
y = nn.Identity()()
q_hid = nn.ReLU(true)(nn.Linear(x_dim+y_dim,hid_dim)(nn.JoinTable(2){x,y}))
for i=1,infer_layers-1 do
    q_hid = nn.ReLU(true)(nn.Linear(hid_dim,hid_dim)(q_hid))
end
q_mu = nn.Linear(hid_dim,z_dim)(q_hid)
q_log_sig = nn.Linear(hid_dim,z_dim)(q_hid)
inference_network = nn.gModule({x,y},{q_mu,q_log_sig})
--generative network (decoder)------------------------
x = nn.Identity()()
z = nn.Identity()()
g_hid = nn.ReLU(true)(nn.Linear(x_dim+z_dim,hid_dim)(nn.JoinTable(2){x,z}))
for i=1,gen_layers-1 do
    g_hid = nn.ReLU(true)(nn.Linear(hid_dim,hid_dim)(g_hid))
end
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
z = nn.Sampler()(p_z)
g_y = generative_network{x,z}
sampling_network = nn.gModule({x},{g_y})



w,dw = network:getParameters()
print(w:numel())
o = network:forward{torch.rand(13,x_dim),torch.rand(13,y_dim)}
network:backward({torch.rand(13,x_dim),torch.rand(13,y_dim)},{torch.zeros(13,y_dim),torch.zeros(1)})

y_data = torch.eye(x_dim)
right_data = torch.zeros(x_dim,x_dim)
right_data[1]:copy(y_data[-1])
right_data[{{2,-1}}]:copy(y_data[{{1,-2}}])
left_data = torch.zeros(x_dim,x_dim)
left_data[{{1,-2}}]:copy(y_data[{{2,-1}}])
left_data[-1]:copy(y_data[1])
right = torch.zeros(x_dim,2)
right[{{},2}] = 1
left = torch.zeros(x_dim,2)
left[{{},1}] = 1
x_data = right_data:cat(left_data,1)
y_data = y_data:cat(right):cat(y_data:cat(left),1)


full_x_data = x_data:clone()
full_y_data = y_data:clone()

--x_data[3]:zero():add(1/x_dim)
--y_data[3]:zero():add(1/y_dim)
crit = nn.BCECriterion()
--[[mb_dim = 12
x_mb = torch.zeros(mb_dim,x_dim)
y_mb = torch.zeros(mb_dim,y_dim)
--]]
function train(w)
    network:zeroGradParameters()
    if mb_dim then
        shuf = torch.randperm(x_data:size(1))
        for i=1,mb_dim do
            x_mb[i] = x_data[shuf[i] ]
            y_mb[i] = y_data[shuf[i] ]
        end
    else
        x_mb = x_data
        y_mb = y_data
    end
        
    output,loss = unpack(network:forward{x_mb,y_mb})
    loss = loss + crit:forward(output,y_mb)
    grad = crit:backward(output,y_mb)
    network:backward({x_mb,y_mb},{grad,torch.zeros(1)})
    return loss,dw
end
num_steps = 1e6
refresh = 1e3
config = {learningRate=1e-3}
cumloss = 0
lower_bound = torch.zeros(num_steps/refresh)
for t=1,num_steps do
    _,batchloss = optim.adam(train,w,config)
    cumloss = cumloss + batchloss[1]
    if t % refresh == 0 then
        print(t,cumloss/refresh,w:norm(),dw:norm(),sys.clock()-time)
        samples = sampling_network:forward(full_x_data)
        gnuplot.figure(1)
        gnuplot.imagesc(samples)
        z = inference_network:forward{full_x_data,samples}
        gnuplot.figure(2)
        gnuplot.bar(z[2])
        --gnuplot.bar(prior_network.output[2])
        gnuplot.figure(3)
        lower_bound[t/refresh] = cumloss/refresh
        gnuplot.plot(lower_bound[{{1,t/refresh}}])
        cumloss = 0
        time = sys.clock()
    end
end


