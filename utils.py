import cv2
import numpy as np

import torch
import torch.nn.functional as F

from typing import Dict, List

distribute_attribute_list = ["where", "what", "when"]

def to_cpu(x:torch.Tensor):
    return x.cpu()

def detach(x:torch.Tensor):
    return x.detach()

def detach_to_cpu(x:torch.Tensor):
    return x.detach().cpu()

def to_bfloat16(x:torch.Tensor):
    return x.bfloat16()

def to_float(x:torch.Tensor):
    return x.float()

def traverse_apply(x, func):
    if isinstance(x, Dict):
        return {k:traverse_apply(x[k], func) for k in x.keys()}
    elif isinstance(x, torch.Tensor):
        return func(x)
    else:
        raise NotImplementedError

def merge(x:List):
    data = {}
    for k in x[0].keys():
        if isinstance(x[0][k], torch.Tensor):
            data[k] = torch.cat([item[k][None] for item in x], 0)
        elif isinstance(x[0][k], Dict):
            data[k] = merge([item[k] for item in x])
        else:
            raise NotImplementedError
    return data

def heatMapPlot(mem, annot, test_mode=False):
    mem = mem / (mem.norm(p=2, dim=-1, keepdim=True) + 1e-8)
    sim = mem.transpose(0,1) # B T embsize

    test_mode = True
    repeat_width = 30 if test_mode else 5

    blank_width = 2

    annot_all = torch.cat([annot.cpu(), torch.zeros(annot.shape[0], blank_width)], 1).flatten().numpy()
    annot_all *= 255 // annot_all.max()
    annot_all = annot_all[None].repeat(repeat_width,0)
    annot_all = cv2.applyColorMap(annot_all.astype(np.uint8), cv2.COLORMAP_HOT)
    sim_all = torch.cat([sim, torch.zeros(sim.shape[0], blank_width, sim.shape[2]).to(sim.device)], 1).flatten(0,1)
    sim_all = (sim_all @ sim_all.transpose(-1, -2)).detach().cpu().numpy()
    sim_all -= sim_all.min()
    sim_all /= (sim_all.max() + 1e-8)
    sim_all *= 255
    img_all = cv2.applyColorMap(sim_all.astype(np.uint8), cv2.COLORMAP_JET)
    img_all = np.concatenate([annot_all, img_all], 0)
    
    img_list = []
    sim = (sim @ sim.transpose(-1, -2)).detach().cpu().numpy()
    sim -= sim.min()
    sim /= (sim.max() + 1e-8)
    sim *= 255
    annot = annot.detach().cpu().numpy()
    annot *= 255 // annot.max()
    for i in range(sim.shape[0]):
        annot_i = cv2.applyColorMap(annot[i][None].repeat(repeat_width,0).astype(np.uint8), cv2.COLORMAP_HOT)
        img_i = cv2.applyColorMap(sim[i].astype(np.uint8), cv2.COLORMAP_JET)
        img_i = np.concatenate([annot_i, img_i], 0)
        img_list.append(img_i)
    return img_all, img_list

def batch2cuda(batch):
    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].cuda()
        elif isinstance(batch[k], Dict):
            batch[k] = batch2cuda(batch[k])
    return batch

def attribute_merge(data, attribute_list):
    return {"whole-event" : torch.cat([data[attribute] for attribute in attribute_list], -1)}

def attribute_divide(data, attribute_list):
    data = torch.chunk(data["whole-event"], chunks=len(attribute_list), dim=-1)
    return {attribute_list[i]:data[i] for i in range(len(attribute_list))}

def batch_prepare(batch, mode=None, attribute_list=None):
    batch = batch2cuda(batch)
    
    vision_sensory_input = batch['video_feat'].transpose(0, 1)
    text_sensory_input = batch['text_feat'].transpose(0, 1)

    curEvent = {}
    nxtEvent = {}
    curEvent['where'] = batch['where_feat'].transpose(0, 1)
    curEvent['what']  = batch['what_feat'].transpose(0, 1)
    curEvent['when'] = batch['when_feat'].transpose(0, 1)
    nxtEvent['where'] = batch['next_where_feat'].transpose(0, 1)
    nxtEvent['what']  = batch['next_what_feat'].transpose(0, 1)
    nxtEvent['when']  = batch['next_when_feat'].transpose(0, 1)

    if mode == 'merge':
        curEvent = attribute_merge(curEvent, attribute_list)
        nxtEvent = attribute_merge(nxtEvent, attribute_list)

    plan = batch['plan'].transpose(0, 1)

    annot = batch['what_annot'].cpu()
    annot_where = batch['where_annot'].cpu()
    annot_when = batch['when_annot'].cpu()
    annot_action = batch['action_annot'].cpu()

    return vision_sensory_input, text_sensory_input, curEvent, nxtEvent, plan, annot, annot_where, annot_when, annot_action

def outputs_prepare(outputs, mode=None, attribute_list=None):
    if outputs is None:
        return outputs
    if mode == 'merge':
        for k in ['episodic memory', 'event prediction', 'next event prediction', 'episodic memory target']:
            outputs[k] = attribute_divide(outputs[k], attribute_list)
    return outputs

def loss_process(loss):
    loss_sum = 0.
    loss_info = ''
    for k in loss.keys():
        loss_sum = loss_sum + loss[k]
        loss_info += f'  {k} : {loss[k].item()}\n'
    loss_sum = loss_sum / len(loss.keys())
    loss_info += f'  average loss : {loss_sum.item()}\n'
    return loss_sum, loss_info

def eventDict2Tensor(event, attribute_list):
    return torch.cat([event[attribute] for attribute in attribute_list], -1)

def Metrics(X, Y):
    l1_loss = F.l1_loss(X, Y)
    l2_loss = F.mse_loss(X, Y)
    std_x = torch.std(X, dim=-1, keepdims=True)
    std_y = torch.std(Y, dim=-1, keepdims=True)
    corr = ((X - X.mean(-1, keepdims=True)) * (Y - Y.mean(-1, keepdims=True)) / std_x / std_y).mean()
    sim = ((X / X.norm(dim=-1, keepdim=True)) * (Y / Y.norm(dim=-1, keepdim=True))).sum(-1).mean()
    return {
        'l1_loss' : l1_loss.item(),
        'l2_loss' : l2_loss.item(),
        'correlation' : corr.item(),
        'similarity' : sim.item(),
    }

@torch.no_grad()
def inference(model, test_data, sigma, exp_dir, filename_tag, save_outputs=False):
    model.eval()
    vision_sensory_input = test_data['vision_sensory_input']
    text_sensory_input = test_data['text_sensory_input']
    curEvent = test_data['curEvent_target']
    nxtEvent = test_data['nxtEvent_target']
    plan = test_data['plan']
    annot = test_data['annot']
    annot_action = test_data['annot_action']

    vision_sensory_input += torch.randn_like(vision_sensory_input, device=vision_sensory_input.device) * (sigma**2)
    text_sensory_input += torch.randn_like(text_sensory_input, device=text_sensory_input.device) * (sigma**2)
    sensory_input = torch.cat([vision_sensory_input, text_sensory_input], -1)

    outputs, loss = model(
        vision_sensory_input=vision_sensory_input,
        text_sensory_input=text_sensory_input,
        curEvent=curEvent,
        nxtEvent=nxtEvent,
        plan=plan,
        need_outputs=True
    )
    outputs = outputs_prepare(outputs, model.mode, distribute_attribute_list)
    if save_outputs:
        torch.save(traverse_apply(outputs, to_bfloat16), exp_dir + f'outputs-{filename_tag}.pt')
    all_metrics = run_metrics(model, test_data, outputs)

    sensory_input = sensory_input[:,:1]
    cur_event = eventDict2Tensor(outputs['event prediction'], distribute_attribute_list)[:,:1]
    wm = outputs['working memory'][:,:1]
    em = eventDict2Tensor(outputs['episodic memory'], distribute_attribute_list)[:,:1]

    annot = annot[:1]
    annot_action = annot_action[:1]

    interval = vision_sensory_input.shape[0] // 1000 # ~ window_size / 1000

    sensory_input = sensory_input[::interval].detach().cpu()
    cur_event = cur_event[::interval].detach().cpu()
    em = em[::interval].detach().cpu()
    annot = annot[:,::interval].detach().cpu()
    wm = wm[::10*interval].detach().cpu()
    annot_action = annot_action[:,::10*interval].detach().cpu()

    img_all, img_list = heatMapPlot(cur_event, annot)
    cv2.imwrite(exp_dir + f'inference-{filename_tag}-sigma{sigma}-training-event.png', img_all)

    img_all, img_list = heatMapPlot(em, annot)
    cv2.imwrite(exp_dir + f'inference-{filename_tag}-sigma{sigma}-training-em.png', img_all)

    img_all, img_list = heatMapPlot(sensory_input, annot)
    cv2.imwrite(exp_dir + f'inference-{filename_tag}-sigma{sigma}-training-sensory_input.png', img_all)

    wm = torch.cat([torch.zeros_like(wm[:,:,:1], device=wm.device), wm], 2)
    wm = wm.flatten(0, 2)[:,None]
    img_all, img_list = heatMapPlot(wm, annot_action.flatten()[:,None].repeat(1,model.num_slots+1).reshape(1, -1))
    cv2.imwrite(exp_dir + f'inference-{filename_tag}-sigma{sigma}-training-wm.png', img_all)

    return all_metrics

def run_metrics(model, test_data, outputs):
    curEvent_target = test_data['curEvent_target']
    curEvent_target = torch.cat([curEvent_target[attribute] for attribute in model.attribute_list], -1)
    nxtEvent_target = test_data['nxtEvent_target']
    nxtEvent_target = torch.cat([nxtEvent_target[attribute] for attribute in model.attribute_list], -1)
    curEvent_pred   = outputs['event prediction']
    curEvent_pred   = torch.cat([curEvent_pred[attribute] for attribute in distribute_attribute_list], -1).float()
    nxtEvent_pred   = outputs['next event prediction']
    nxtEvent_pred   = torch.cat([nxtEvent_pred[attribute] for attribute in distribute_attribute_list], -1).float()

    sensory_input = test_data['sensory_input']
    sensory_predict = outputs['sensory input prediction'].float()

    all_metrics = {
        'curEvent' : Metrics(curEvent_target.cpu(), curEvent_pred),
        'nxtEvent' : Metrics(nxtEvent_target.cpu(), nxtEvent_pred),
        'sensory' : Metrics(sensory_input.cpu(), sensory_predict),
    }

    print('----- curEvent target vs. prediction -----')
    print(all_metrics['curEvent'])

    print('----- nxtEvent target vs. prediction -----')
    print(all_metrics['nxtEvent'])

    print('----- sensory target vs. prediction -----')
    print(all_metrics['sensory'])

    return all_metrics

def run_metrics_attractor(model, test_data):
    curEvent_target = test_data['curEvent_target']
    curEvent_target = torch.cat([curEvent_target[attribute] for attribute in model.attribute_list], -1)

    curEvent_target_unique = torch.unique(curEvent_target, dim=0)
    curEvent_target_unique = model.module_dict['episodic memory'].Tensor2dict(curEvent_target_unique)
    EM_target = model.module_dict['events'].event2em(curEvent_target_unique)

    for attribute in model.attribute_list:
        EM_target[attribute] = EM_target[attribute].flatten(0, -2)

    new_state = {attribute:EM_target[attribute] for attribute in model.attribute_list}

    for i in range(10000):
        new_state = model.module_dict['episodic memory'].attractor_iteration(new_state)

    print('----------------------- test -----------------------')
    delta = 0.
    thre = 1e-3
    for attribute in model.attribute_list:
        delta_attribute = (new_state[attribute] - EM_target[attribute]).abs().sum(-1)
        delta = delta + delta_attribute

        if model.mode != "merge":
            print(f'{attribute} acc : {(1.0 * (delta_attribute < thre)).mean()}')
    if model.mode == "merge":
        merged_target = EM_target['whole-event']
        merged_state = new_state['whole-event']
    else:
        merged_target = torch.cat([EM_target[attribute] for attribute in model.attribute_list], -1)
        merged_state = torch.cat([new_state[attribute] for attribute in model.attribute_list], -1)

    print(f'whole event acc : {(100.0 * (delta < thre)).mean()}')
    print(f'whole delta : {delta.mean()}')
    print(f'#attractor before iteration : {torch.unique(merged_target, dim=0).shape[0]}')
    print(f'#attractor after iteration : {torch.unique(merged_state, dim=0).shape[0]}')
