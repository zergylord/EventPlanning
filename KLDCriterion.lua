require 'nn'

local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Module')

function KLDCriterion:updateOutput(input)
   -- KL(input[2] || input[1])
   local mu1 = input[2][1]:clone()
   local logv1 = input[2][2]:clone()
   local mu2 = input[1][1]:clone()
   local logv2 = input[1][2]:clone()

   local v1 = torch.exp(logv1)
   local v2 = torch.exp(logv2)
   
   self.output = (torch.add(logv2, -logv1):add(-1):addcdiv(v1, v2):
                     addcdiv((mu2 - mu1):pow(2), v2))

   return self.output:sum() * 0.5
end

function KLDCriterion:updateGradInput(input, gradOutput)
   -- KL(input[2] || input[1])
   local mu1 = input[2][1]:clone()
   local logv1 = input[2][2]:clone()
   local mu2 = input[1][1]:clone()
   local logv2 = input[1][2]:clone()
   
   local v1 = torch.exp(logv1)
   local v2 = torch.exp(logv2)

   local diff12 = mu1:add(-mu2)
   local dmu1 = torch.cdiv(diff12, v2)
   local dmu2 = torch.cdiv(-diff12, v2)
   local div12 = torch.cdiv(v1, v2)
   local dlogv1 = div12:clone():add(-1):div(2)
   -- be careful: use of inplace
   local dlogv2 = div12:mul(-1):add(1):add(-diff12:pow(2):cdiv(v2)):div(2)

   -- return grad w.r.t. input first
   self.gradInput = {{dmu2, dlogv2}, {dmu1, dlogv1}}
   return self.gradInput
end

