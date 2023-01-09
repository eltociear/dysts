#!/usr/bin/python

import os
import numpy as np
import torch
import json

import torch.optim as optim
from torchdiffeq import odeint

import dysts
import dysts.flows
from dysts.base import get_attractor_list


from resources.node import ODEFunc
from resources.node_utils import BatchLoader, get_train_test

niters = 500

cwd = os.path.dirname(os.path.realpath(__file__))
# cwd = os.getcwd()
input_path = os.path.dirname(cwd)  + "/dysts/data/train_multivariate__pts_per_period_100__periods_12.json"
output_path = cwd + "/results/results_neural_ode_multivariate.json"

equation_data = load_file(input_path)

try:
    with open(output_path, "r") as file:
        all_results = json.load(file)
        print("Existing database found")
except FileNotFoundError:
    all_results = dict()

# from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
# score_func = lambda x, y: mean_absolute_percentage_error(x, y, symmetric=True)
def smape(x, y):
    """Symmetric mean absolute percentage error"""
    return 100 * np.mean(np.abs(x - y) / (np.abs(x) + np.abs(y))) * 2
score_func = smape

for equation_name in get_attractor_list():


    print(equation_name, flush=True)
    if equation_name in all_results.keys():
        print("Skipped")
        continue

    train_data = np.copy(np.array(equation_data.dataset[equation_name]["values"]))

    eq.ic = train_ic
    tpts_train, sol_train = eq.make_trajectory(
        1000, resample=True, return_times=True, **kwargs
    )
    eq.ic = test_ic
    tpts_test, sol_test = eq.make_trajectory(
        200, resample=True, return_times=True, **kwargs
    )
    

    # (t_train, sol_train), (t_test, sol_test) = get_train_test(
    #     equation_name, pts_per_period=100
    # )

    bt = BatchLoader(sol_train, 30, tpts=t_train, batch_size=128)
    
    try:
        func = ODEFunc(sol_train.shape[-1]).to(bt.device)
        optimizer = optim.Adam(func.parameters(), lr=1e-2)
        #optimizer = optim.SGD(func.parameters(), lr=1e-3)

        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = bt.get_batch()
        pred_y = odeint(func, batch_y0, batch_t).to(bt.device)

        loss_history = list()
        for itr in range(1, niters + 1):
            optimizer.zero_grad()
            batch_y0, batch_t, batch_y = bt.get_batch()
            pred_y = odeint(func, batch_y0, batch_t).to(bt.device)
            loss = torch.mean(torch.square(pred_y - batch_y))  # MSE
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
    except AssertionError:
        print("Integration error encountered, skipping this entry for now")
        continue
        
    # might need to pass batches not single
    sol_pred = odeint(func, 
                  torch.from_numpy(sol_test[0].astype(np.float32)),
                  torch.from_numpy(t_test.astype(np.float32))
                 ).to(bt.device)
    sol_pred = sol_pred.detach().numpy()
    
    score_val = score_func(sol_test, sol_pred)
    
    all_results[equation_name] = dict()
    all_results[equation_name]["smape"] = score_val
    all_results[equation_name]["traj_true"] = sol_test.tolist()
    all_results[equation_name]["traj_pred"] = sol_pred.tolist()



    print(equation_name, score_val, flush=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=4, sort_keys=True)   
        



