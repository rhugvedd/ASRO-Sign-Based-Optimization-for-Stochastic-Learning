import torch
from torch.optim.optimizer import Optimizer, required

class CustomAdam(Optimizer):
    def __init__(self, params, lr=required, betas=(0.9, 0.999), eps=1e-8):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(CustomAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('CustomAdam does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                denom = corrected_exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']

                p.data.addcdiv_(corrected_exp_avg, denom, value=-step_size)
                
        return loss

class Asro(Optimizer):
    def __init__(self, params, lr=required, decrement=required, min_lr_scale_clamp = required, decr_start_step = 0, betas=(0.9, 0.999), eps=1e-8):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        self.decrement = decrement
        self.min_lr_scale_clamp = min_lr_scale_clamp
        self.decr_start_step = decr_start_step

        print(f"Min LR Scale Clamp : {self.min_lr_scale_clamp}")
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(Asro, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Asro does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['lr_scaler'] = torch.ones_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                prev_exp_avg_signs = torch.sign(exp_avg)

                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if state['step'] >= self.decr_start_step:
                    sign_change = torch.abs(torch.sign(exp_avg) - prev_exp_avg_signs)
                    state['lr_scaler'] -= (sign_change == 2) * self.decrement

                    state['lr_scaler'] = torch.clamp(state['lr_scaler'], min=self.min_lr_scale_clamp, max = 1)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                denom = corrected_exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']

                p.data.addcdiv_(corrected_exp_avg * state['lr_scaler'], denom, value=-step_size)

        return loss

class AccAsroFinalScale(Optimizer):
    def __init__(self, params, lr=required, increment=required, decrement=required, max_lr_scale_clamp=required, min_lr_scale_clamp=required, decr_start_step=required, num_iters=required, betas=(0.9, 0.999), eps=1e-8):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        self.max_lr_scale_clamp = max_lr_scale_clamp
        self.min_lr_scale_clamp = min_lr_scale_clamp
        self.decrement = decrement
        self.increment = increment
        self.decr_start_step = decr_start_step
        self.num_iters = num_iters

        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(AccAsroFinalScale, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AccAsroFinalScale does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['lr_scaler'] = torch.ones_like(p.data)
                    state['lr_scaler_buf'] = torch.ones_like(p.data)
                    state['prev_grad'] = torch.zeros_like(grad)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                prev_exp_avg_signs = torch.sign(exp_avg)
                prev_grad_signs = torch.sign(state['prev_grad'])

                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if state['step'] >= self.decr_start_step:
                    exp_avg_sign_change = torch.abs(torch.sign(exp_avg) - prev_exp_avg_signs)
                    grad_sign_change = torch.abs(torch.sign(grad) - prev_grad_signs)

                    state['lr_scaler'][grad_sign_change == 2] = state['lr_scaler_buf'][grad_sign_change == 2]

                    state['lr_scaler'] -= (exp_avg_sign_change == 2) * self.decrement
                    
                    state['lr_scaler_buf'][exp_avg_sign_change == 2] = state['lr_scaler'][exp_avg_sign_change == 2]
                    
                    state['lr_scaler'] += (grad_sign_change <= 1) * self.increment * (0.01 + 1 - (state['step'] / self.num_iters))

                    state['lr_scaler'] = torch.clamp(state['lr_scaler'], min=self.min_lr_scale_clamp, max=self.max_lr_scale_clamp)

                state['prev_grad'] = grad.detach().clone()

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                denom = corrected_exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']

                p.data.addcdiv_(corrected_exp_avg * state['lr_scaler'], denom, value=-step_size)

        return loss
