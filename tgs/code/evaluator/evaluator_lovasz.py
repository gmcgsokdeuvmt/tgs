from evaluator import config_evaluator
import torch
eval_functions = config_evaluator.config_lovasz

def evaluate(model, input_x, input_t):
    logit = model(input_x)
    result = { 
        key : eval_functions[key](logit, input_t)
        for key in  eval_functions
    }
    return result

config_coef = config_evaluator.config_lovasz_coef
def calc_total_loss(eval_result):
    total_loss = torch.mean(
        sum([
            config_coef[key] * eval_result[key] 
            for key in config_coef
        ]) 
    )
    return total_loss