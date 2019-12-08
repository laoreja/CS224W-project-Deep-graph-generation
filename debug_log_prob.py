import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

B = 1
K = 3
E = 4

logsoftmax = nn.LogSoftmax(dim=-1)


theta_logit = torch.rand(B, E, K)
alpha_logit = torch.rand(B, E, K)

# correct
theta = F.logsigmoid(theta_logit)
print('theta', theta.size())


alpha_sum = alpha_logit.mean(dim=1)
print(alpha_sum.size())
log_alpha = logsoftmax(alpha_sum)
print('log_alpha', log_alpha.size())

log_prob = torch.logsumexp(log_alpha[:, None, :] + theta, dim=2)
print('log prob', log_prob)
print('prob', log_prob.exp())

total_log_prob = log_prob.mean()
print('total_log_prob', total_log_prob)

# in the code
theta_logit2 = theta_logit.view(-1, K)
log_theta2 = F.logsigmoid(theta_logit2)

alpha_logit2 = alpha_logit.view(-1, K)
logsoftmax2 = nn.LogSoftmax(dim=1)
log_alpha2 = logsoftmax2(alpha_logit2)
log_alpha2 = log_alpha2.view(B, E, K)
log_alpha2 = log_alpha2.mean(dim=1)
#print('log_theta2', log_theta2)
#print('log_alpha2', log_alpha2)
log_prob = log_theta2 + log_alpha2

#print('log_theta2 reshape', log_theta2.view(B, E, K))
#print('log_alpha2 reshape', log_alpha2[:, None, :])
#print('sum', log_theta2.view(B, E, K) + log_alpha2[:, None, :])

print('log_prob 2', log_prob)
log_prob = torch.logsumexp(log_prob, dim=1)
print('log_prob 2', log_prob)
print('prob 2', log_prob.exp())
total = log_prob.sum() / log_theta2.size(0) / B

print('total2', total)


# in the code
theta_logit2 = theta_logit.view(-1, K)
log_theta2 = F.logsigmoid(theta_logit2)

log_alpha3 = logsoftmax2(alpha_logit.mean(dim=1))

#alpha_logit2 = alpha_logit.view(-1, K)
#logsoftmax2 = nn.LogSoftmax(dim=1)
#log_alpha2 = logsoftmax2(alpha_logit2)
#log_alpha2 = log_alpha2.view(B, E, K)
#log_alpha2 = log_alpha2.mean(dim=1)
#print('log_theta2', log_theta2)
#print('log_alpha2', log_alpha2)
log_prob = log_theta2 + log_alpha3

#print('log_theta2 reshape', log_theta2.view(B, E, K))
#print('log_alpha2 reshape', log_alpha2[:, None, :])
#print('sum', log_theta2.view(B, E, K) + log_alpha2[:, None, :])

#print('log_prob 3', log_prob)
log_prob = torch.logsumexp(log_prob, dim=1)
print('log_prob 3', log_prob)
print('prob 3', log_prob.exp())
total = log_prob.sum() / log_theta2.size(0) / B

print('total3', total)


print(np.arange(0.5, 1, 0.1))

from decimal import Decimal
a = 0.0007042749154120287
print('%.2E' % Decimal(a))
print('{:.2E}'.format(Decimal(a)))