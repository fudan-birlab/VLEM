import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import *

class CrossAttention(nn.Module):

    def __init__(self, dim_q, dim_kv, output_size):
        super().__init__()
        self.fc_q = nn.Linear(dim_q,  output_size, bias=False)
        self.fc_k = nn.Linear(dim_kv, output_size, bias=False)
        self.fc_v = nn.Linear(dim_kv, output_size, bias=False)
        self.softmax_scale = output_size ** 0.5
        self.softmax = nn.Softmax(dim=-1)
        self.activation = nn.Identity()

    def forward(self, x_q, x_kv):
        q = self.fc_q(x_q)
        k = self.fc_k(x_kv)
        v = self.fc_v(x_kv)
        attn_weights = self.softmax(q @ k.transpose(-1, -2) / self.softmax_scale)

        return self.activation(attn_weights @ v)

class SensoryInput(nn.Module):

    def __init__(self, vision_size, text_size, wm_size):
        super().__init__()
        
        self.input_size = vision_size + text_size
        self.wm_size = wm_size
        self.input2wm = nn.Sequential(
            nn.Linear(self.input_size, wm_size, bias=False),
            nn.Tanh()
        )
        self.wm2input = nn.Linear(wm_size, self.input_size, bias=False)
    
    def reset(self, *kwargs):
        self.sensory_input = None

    def forward(self, vision_input, text_input):
        self.sensory_input = torch.cat([vision_input, text_input], -1)
        return self.sensory_input

    def backward_loss(self, wm_feat):
        # sensory_input : ... * N1
        # wm_feat : ... * num_slots * N2
        wm_target = self.input2wm(self.sensory_input)
        with torch.no_grad():
            all_input_pred = self.wm2input(wm_feat.detach())
            indices = ((wm_target.unsqueeze(-2).detach() - wm_feat.detach()) ** 2).sum(dim=-1, keepdims=True).min(axis=-2).indices.unsqueeze(-2).expand(-1, -1, self.wm_size)
            assert len(indices.shape) == 3 # make sure expanded dimension is the last dimension
        wm_feat_chosen = torch.gather(wm_feat, -2, indices).squeeze(-2)
        input_pred = self.wm2input(wm_feat_chosen)
        loss = {
            'loss sensory input' : F.mse_loss(input_pred, self.sensory_input),
            'loss chosen working memory' : F.mse_loss(wm_feat_chosen, wm_target)
        }
        return all_input_pred, input_pred, loss

class WorkingMemory(nn.Module):

    def __init__(self, input_size, hidden_size, num_slots):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_slots = num_slots

        self.memorySlots = nn.RNNCell(input_size=input_size, hidden_size=self.num_slots*self.hidden_size, bias=True, nonlinearity='tanh')

    def step_contrastive_loss(self):
        return self.contrastive_loss(self.hidden_state.reshape(1, -1, self.num_slots*self.hidden_size))

    def contrastive_loss(self, hidden_states):
        T, B, N = hidden_states.shape
        assert N == self.num_slots * self.hidden_size
        slots_states = hidden_states.reshape(T, B, self.num_slots, self.hidden_size)
        # only calculate active slots, let norm be activity of slot
        slots_states = slots_states / (self.hidden_size ** 0.5)
        sim = (slots_states @ slots_states.transpose(-1, -2)) * (1. - torch.eye(self.num_slots, device=slots_states.device)[None,None])
        return sim.abs().mean() * self.num_slots / (self.num_slots - 1)

    def reset(self, batch_size, device):
        self.hidden_state = torch.rand(batch_size, self.num_slots * self.hidden_size, device=device) * 2. - 1.

    def step_forward(self, sensory_input):
        B, N = sensory_input.shape
        assert N == self.input_size
        self.hidden_state = self.memorySlots(sensory_input, self.hidden_state)
        return self.hidden_state.reshape(B, self.num_slots, self.hidden_size)

    def forward(self, sensory_input):
        T, B, N = sensory_input.shape
        assert N == self.input_size
        hidden_state = torch.rand(B, self.num_slots * self.hidden_size, device=sensory_input.device) * 2. - 1.
        hidden_state_list = []
        for t in range(T):
            hidden_state = self.memorySlots(sensory_input[t], hidden_state)
            hidden_state_list.append(hidden_state[None])
        return torch.cat(hidden_state_list, 0).reshape(T, B, self.num_slots, self.hidden_size) # T * B * num_slots * hidden_size

class Entorhinal(nn.Module):

    def __init__(self, event_size, wm_size, ento_size):
        super().__init__()
        self.event_size = event_size
        self.wm_size = wm_size
        self.ento_size = ento_size
        self.crossAttn = CrossAttention(dim_q=event_size, dim_kv=wm_size, output_size=ento_size)
    
    def reset(self, *kwargs):
        pass

    def forward(self, current_event, wm):
        current_event = current_event[:, None]
        assert len(current_event.shape) == len(wm.shape) and current_event.shape[-1] == self.event_size and wm.shape[-1] == self.wm_size
        return self.crossAttn(x_q=current_event, x_kv=wm)[:,0]

class Event(nn.Module):

    def __init__(self, event_size, ento_size, em_size, plan_size):
        super().__init__()

        self.event_size = event_size
        self.ento_size = ento_size
        self.em_size = em_size
        self.event2em = nn.Sequential( # curEvent-to-attractor
            nn.Linear(event_size, em_size, bias=False),
            nn.Tanh()
        )

        self.em2curEvent = nn.Linear(em_size+ento_size, event_size, bias=False)
        self.em2nxtEvent = nn.Linear(em_size+plan_size, event_size, bias=False)

    def reset(self, batch_size, device):
        self.curEvent_pred = torch.rand(batch_size, self.event_size, device=device) * 2. - 1.
        self.curEvent_feat = torch.rand(batch_size, self.event_size, device=device) * 2. - 1.
        self.nxtEvent_feat = torch.rand(batch_size, self.event_size, device=device) * 2. - 1.
        self.nxtEM_feat = torch.rand(batch_size, self.em_size, device=device) * 2. - 1.
        self.all_curEvent_feat = []

    def forward(self, curEvent_feat, nxtEvent_feat):
        # curEvent_feat : B * N
        B, N = curEvent_feat.shape
        self.curEvent_feat = curEvent_feat
        self.nxtEvent_feat = nxtEvent_feat
        with torch.no_grad():
            self.all_curEvent_feat.append(curEvent_feat[None])
        return self.curEvent_feat, self.nxtEvent_feat

    def contrastive_loss(self):
        all_curEvent_feat = torch.cat(self.all_curEvent_feat, 0)
        loss = torch.tensor(0., device=all_curEvent_feat.device)
        validEvents = torch.unique(all_curEvent_feat.flatten(0, -2), dim=0)
        num_events, N = validEvents.shape
        if num_events > 1:
            validEM = self.event2em(validEvents)
            validEM = validEM / validEM.norm(p=2, dim=-1, keepdim=True)
            sim = (validEM @ validEM.transpose(0, 1)) * (1. - torch.eye(num_events, device=validEM.device))
            loss = sim.mean() * num_events / (num_events - 1)
        return loss

    def backward_loss(self, em_feat, ento_feat, plan_feat):
        # em_feat : B * N1
        # ento_feat : B * N2
        # plan_feat : B * N3
        curEM_target = self.event2em(self.curEvent_feat)

        self.curEvent_pred = self.em2curEvent(torch.cat([em_feat, ento_feat], -1))
        nxtEvent_pred = self.em2nxtEvent(torch.cat([em_feat, plan_feat], -1))

        loss = {
            'loss episodic memory target' : -curEM_target.abs().mean(),
            'loss current episodic memory' : F.l1_loss(curEM_target, em_feat),
            'loss current event' : F.mse_loss(self.curEvent_feat, self.curEvent_pred),
            'loss next event' : F.mse_loss(self.nxtEvent_feat, nxtEvent_pred),
        }
        return self.curEvent_pred, nxtEvent_pred, curEM_target, loss

class Events(nn.Module):

    def __init__(self, attribute_list, event_size, ento_size, em_size, plan_size):
        super().__init__()
        
        self.attribute_list = attribute_list
        self.events = nn.ModuleDict(
            {
                attribute:Event(event_size=event_size, ento_size=ento_size, em_size=em_size, plan_size=plan_size)
                for attribute in attribute_list
            }
        )

    def reset(self, batch_size, device):
        for attribute in self.attribute_list:
            self.events[attribute].reset(batch_size=batch_size, device=device)
        self.curEvent_feat = {attribute:self.events[attribute].curEvent_feat for attribute in self.attribute_list}
        self.nxtEvent_feat = {attribute:self.events[attribute].nxtEvent_feat for attribute in self.attribute_list}
        self.curEvent_pred = {attribute:self.events[attribute].curEvent_pred for attribute in self.attribute_list}

    def forward(self, curEvent_feat:Dict, nxtEvent_feat:Dict):
        for attribute in self.attribute_list:
            self.curEvent_feat[attribute], self.nxtEvent_feat[attribute] = self.events[attribute](curEvent_feat=curEvent_feat[attribute], nxtEvent_feat=nxtEvent_feat[attribute])
        return self.curEvent_feat, self.nxtEvent_feat

    def contrastive_loss(self):
        loss = 0.
        for attribute in self.attribute_list:
            loss = loss + self.events[attribute].contrastive_loss()
        loss = loss / len(self.attribute_list)
        return loss

    @torch.no_grad()
    def event2em(self, event_dict):
        em_dict = {}
        for attribute in self.attribute_list:
            em_dict[attribute] = self.events[attribute].event2em(event_dict[attribute])
        return em_dict

    def backward_loss(self, em_feat:Dict, ento_feat, plan_feat):
        loss = None
        curEM_target = {}
        nxtEvent_pred = {}
        for attribute in self.attribute_list:
            self.curEvent_pred[attribute], nxtEvent_pred[attribute], curEM_target[attribute], loss_event = self.events[attribute].backward_loss(em_feat=em_feat[attribute], ento_feat=ento_feat, plan_feat=plan_feat)
            if loss is None:
                loss = loss_event
            else:
                for k in loss.keys():
                    loss[k] = loss[k] + loss_event[k]
        for k in loss.keys():
            loss[k] = loss[k] / len(self.attribute_list)
        return self.curEvent_pred, nxtEvent_pred, curEM_target, loss

class EpisodicMemory(nn.Module):

    def __init__(self, attribute_list, ento_size, em_size, event_size, drop, additional_iteration=0):
        super().__init__()

        self.ento_size = ento_size
        self.em_size = em_size
        self.event_size = event_size

        self.attribute_list = attribute_list

        self.drop = drop
        self.input_dropout = nn.Dropout(drop)

        self.attractor_RNN = nn.ModuleDict({
            attribute : nn.RNNCell(input_size=self.ento_size+(len(self.attribute_list)-1)*self.em_size, hidden_size=self.em_size, bias=False, nonlinearity='tanh')
            for attribute in self.attribute_list
        })

        self.additional_iteration = additional_iteration

    def reset(self, batch_size, device):
        self.neuron_v = {
            neuron : torch.rand(batch_size, self.em_size, device=device) * 2. - 1.
            for neuron in self.attribute_list
        }

    def step_forward(self, ento_feat, return_dict=True):
        B, N = ento_feat.shape
        ento_feat = self.input_dropout(ento_feat)
        assert N == self.ento_size
        neuron_input = {
            neuron : torch.cat([ento_feat] + [self.neuron_v[pre] for pre in self.attribute_list if pre != neuron], -1)
            for neuron in self.attribute_list
        }
        for neuron in self.attribute_list:
            self.neuron_v[neuron] = self.attractor_RNN[neuron](neuron_input[neuron], self.neuron_v[neuron])

        for _ in range(self.additional_iteration):
            self.neuron_v = self.attractor_iteration(self.neuron_v)

        if return_dict:
            return {neuron : self.neuron_v[neuron] for neuron in self.attribute_list}
        else:
            return torch.cat([self.neuron_v[neuron] for neuron in self.attribute_list], -1)

    def forward(self, ento_feat):
        T, B, N = ento_feat.shape
        ento_feat = self.input_dropout(ento_feat)
        assert N == self.ento_size
        neuron_v = {
            neuron : torch.rand(B, self.hidden_size).to(ento_feat.device) * 2. - 1.
            for neuron in self.attribute_list
        }
        outputs = {
            neuron : []
            for neuron in self.attribute_list
        }
        for i in range(T):
            neuron_input = {
                neuron : torch.cat([ento_feat[i]] + [neuron_v[pre] for pre in self.attribute_list if pre != neuron], -1)
                for neuron in self.attribute_list
            }
            for neuron in self.attribute_list:
                neuron_v[neuron] = self.attractor_RNN[neuron](neuron_input[neuron], neuron_v[neuron])
            for _ in range(self.additional_iteration):
                neuron_v = self.attractor_iteration(neuron_v)
            for neuron in self.attribute_list:
                outputs[neuron].append(neuron_v[neuron][None])
            
        return {neuron : torch.cat(outputs[neuron], 0) for neuron in self.attribute_list}

    @torch.no_grad()
    def constrain_inter_conn(self, neuron1, neuron2):
        assert neuron1 != neuron2
        idx1 = np.where(np.array(self.attribute_list) == neuron1)[0].item()
        idx2 = np.where(np.array(self.attribute_list) == neuron2)[0].item()
        if idx1 < idx2:
            idx2 -= 1
        else:
            idx1 -= 1
        new_conn = (self.attractor_RNN[neuron1].weight_ih.data[:,self.ento_size+idx2*self.em_size:self.ento_size+(idx2+1)*self.em_size] + self.attractor_RNN[neuron2].weight_ih.data[:,self.ento_size+idx1*self.em_size:self.ento_size+(idx1+1)*self.em_size].transpose(1, 0)) / 2.
        self.attractor_RNN[neuron1].weight_ih.data[:,self.ento_size+idx2*self.em_size:self.ento_size+(idx2+1)*self.em_size] = new_conn
        self.attractor_RNN[neuron2].weight_ih.data[:,self.ento_size+idx1*self.em_size:self.ento_size+(idx1+1)*self.em_size] = new_conn

    @torch.no_grad()
    def constrain_attractor(self):
        for i, neuron1 in enumerate(self.attribute_list):
            self.attractor_RNN[neuron1].weight_hh.data = (self.attractor_RNN[neuron1].weight_hh.data + self.attractor_RNN[neuron1].weight_hh.data.transpose(1, 0)) / 2.
            for neuron2 in self.attribute_list[i+1:]:
                self.constrain_inter_conn(neuron1, neuron2)

    def Tensor2dict(self, x):
        x = torch.chunk(x, chunks=len(self.attribute_list), dim=-1)
        return {self.attribute_list[i]:x[i] for i in range(len(self.attribute_list))}

    def attractor_iteration(self, target:Dict):
        origin_shape = target[self.attribute_list[0]].shape
        target = {k:target[k].reshape(-1, self.em_size) for k in target.keys()}
        neuron_input_withconn = {
            neuron : torch.cat([torch.zeros(target[neuron].shape[0], self.ento_size, device=target[neuron].device)] + [target[pre] for pre in self.attribute_list if pre != neuron], -1)
            for neuron in self.attribute_list
        }
        for neuron in self.attribute_list:
            target[neuron] = self.attractor_RNN[neuron](neuron_input_withconn[neuron], target[neuron]).reshape(*origin_shape)
        return target

    def attractor_loss(self, target:Dict):
        target = {k:target[k].reshape(-1, self.em_size) for k in target.keys()}
        neuron_input_withconn = {
            neuron : torch.cat([torch.zeros(target[neuron].shape[0], self.ento_size, device=target[neuron].device)] + [target[pre] for pre in self.attribute_list if pre != neuron], -1)
            for neuron in self.attribute_list
        }
        neuron_input_withoutconn = {
            neuron : torch.cat([torch.zeros(target[neuron].shape[0], self.ento_size, device=target[neuron].device)] + [torch.zeros_like(target[pre], device=target[neuron].device) for pre in self.attribute_list if pre != neuron], -1)
            for neuron in self.attribute_list
        }
        loss = 0.
        for neuron in self.attribute_list:
            state_next_withconn = self.attractor_RNN[neuron](neuron_input_withconn[neuron], target[neuron])
            state_next_withoutconn = self.attractor_RNN[neuron](neuron_input_withoutconn[neuron], target[neuron])
            loss = loss + F.l1_loss(state_next_withconn, target[neuron])
            loss = loss + F.l1_loss(state_next_withoutconn, target[neuron])
        return loss / len(self.attribute_list)

class EpisodicMemoryHopfield(EpisodicMemory):

    def __init__(self, em_model, patterns):
        super().__init__(em_model.attribute_list, em_model.ento_size, em_model.em_size, em_model.event_size, em_model.drop)

        assert len(em_model.attribute_list) == 1

        patterns = patterns.flatten(0, -2)
        weight = (patterns.transpose(0, 1) @ patterns) / patterns.shape[1] * 10
        self.attractor_RNN[self.attribute_list[0]].weight_hh.data = weight.to(self.attractor_RNN[self.attribute_list[0]].weight_hh.data.device)

class EMModel(nn.Module):

    def __init__(self,
            # sensory input
            vision_input_size=1024,
            text_input_size=1024,

            # Event
            attribute_list = ["where", "what", "when"],
            event_size=1024,

            # working memory
            wm_size=256,
            num_slots=7,

            # Entorihnal
            ento_size=1024,

            # episodic memory
            em_size=1024,
            additional_iteration=2,
            drop=0.,

            # plan
            plan_size=1024,

            # mode
            mode='distribute',
            loss_scale={},
        ):
        super().__init__()

        print(
            f' ----------- model configuration ----------- \n' + 
            f'sensory input\n' +
            f'  vision_input_size : {vision_input_size}\n' +
            f'  text_input_size : {text_input_size}\n' +
            f'event\n' +
            f'  attribute_list : {attribute_list}\n' +
            f'  event_size : {event_size}\n' +
            f'working memory\n' +
            f'  wm_size : {wm_size}\n' +
            f'  num_slots : {num_slots}\n' +
            f'entorhinal\n' +
            f'  ento_size : {ento_size}\n' +
            f'episodic memory\n' +
            f'  em_size : {em_size}\n' +
            f'  additional_iteration : {additional_iteration}\n' +
            f'  drop : {drop}\n' +
            f'plan\n' +
            f'  plan_size : {plan_size}\n' +
            f'mode\n' +
            f'  mode : {mode}\n' +
            f'loss scale\n' +
            f'  dict : {loss_scale}\n' + 
            f' ----------- model configuration ----------- \n'
        )

        self.vision_input_size = vision_input_size
        self.text_input_size = text_input_size
        self.attribute_list = attribute_list
        self.event_size = event_size
        self.wm_size = wm_size
        self.num_slots = num_slots
        self.ento_size = ento_size
        self.em_size = em_size
        self.additional_iteration = additional_iteration
        self.plan_size = plan_size
        self.mode = mode
        self.loss_scale = loss_scale

        assert mode in ['distribute', 'merge']

        self.module_dict = nn.ModuleDict(
            {
                "sensory input"   : SensoryInput(vision_size=vision_input_size, text_size=text_input_size, wm_size=wm_size),
                "working memory"  : WorkingMemory(input_size=vision_input_size + text_input_size, hidden_size=wm_size, num_slots=num_slots),
                "entorhinal"      : Entorhinal(event_size=len(attribute_list)*event_size, wm_size=wm_size, ento_size=ento_size),
                "episodic memory" : EpisodicMemory(attribute_list=attribute_list, ento_size=ento_size, em_size=em_size, event_size=event_size, drop=drop, additional_iteration=additional_iteration),
                "events"          : Events(attribute_list=attribute_list, event_size=event_size, ento_size=ento_size, em_size=em_size, plan_size=plan_size),
            }
        )
        self.step = 0

    def eventDict2Tensor(self, event:Dict):
        return torch.cat([event[attribute] for attribute in self.attribute_list], -1)

    def reset(self, batch_size, device):
        for module_name in self.module_dict.keys():
            self.module_dict[module_name].reset(batch_size, device)

    def constrain_attractor(self):
        self.module_dict['episodic memory'].constrain_attractor()

    def to_hopfield(self, EM_target):
        if len(self.module_dict['episodic memory'].attribute_list) != 1:
            self.module_dict['episodic memory'].em_size *= len(self.module_dict['episodic memory'].attribute_list)
            self.module_dict['episodic memory'].attribute_list = ['whole-event']
        self.module_dict['episodic memory'] = EpisodicMemoryHopfield(self.module_dict['episodic memory'], EM_target).to(EM_target.device)

    def forward(
        self,
        vision_sensory_input,
        text_sensory_input,
        curEvent:Dict,
        nxtEvent:Dict,
        plan,
        need_outputs=False,
    ):
        T, B, N = vision_sensory_input.shape
        self.reset(B, vision_sensory_input.device)
        output_list = []
        loss_enable = ['loss sensory', 'loss events', 'loss contrastive working memory', 'loss episodic memory attractor']
        loss = {}
        for t in range(T):
            curEvent_feat = {k:curEvent[k][t] for k in curEvent.keys()}
            nxtEvent_feat = {k:nxtEvent[k][t] for k in nxtEvent.keys()}

            output = {}
            sensory_input             = self.module_dict['sensory input'](vision_input=vision_sensory_input[t], text_input=text_sensory_input[t])
            curEvent_t, nxtEvent_t    = self.module_dict['events'](curEvent_feat=curEvent_feat, nxtEvent_feat=nxtEvent_feat)

            output['working memory']  = self.module_dict['working memory'].step_forward(sensory_input)
            output['entorhinal']      = self.module_dict['entorhinal'](current_event=self.eventDict2Tensor(self.module_dict['events'].curEvent_pred), wm=output['working memory'])
            output['episodic memory'] = self.module_dict['episodic memory'].step_forward(ento_feat=output['entorhinal'])

            output['all sensory input prediction'], output['sensory input prediction'], output['loss sensory'] = self.module_dict['sensory input'].backward_loss(output['working memory'])
            output['event prediction'], output['next event prediction'], output['episodic memory target'], output['loss events'] = self.module_dict['events'].backward_loss(em_feat=output['episodic memory'], ento_feat=output['entorhinal'], plan_feat=plan[t])

            output['loss contrastive working memory']  = self.module_dict['working memory'].step_contrastive_loss()
            output['loss episodic memory attractor'] = self.module_dict['episodic memory'].attractor_loss(output['episodic memory target'])

            for loss_name in loss_enable:
                if isinstance(output[loss_name], Dict):
                    for k in output[loss_name].keys():
                        assert k not in loss_enable
                        loss[k] = (loss[k] if k in loss.keys() else 0.) + output[loss_name][k]
                else:
                    loss[loss_name] = (loss[loss_name] if loss_name in loss.keys() else 0.) + output[loss_name]

            if need_outputs:
                output_list.append(traverse_apply(output, detach_to_cpu))

        if not need_outputs: # training
            self.step += 1
        for loss_name in loss.keys():
            loss[loss_name] = loss[loss_name] / T
        loss['loss contrastive episodic memory'] = self.module_dict['events'].contrastive_loss()

        for loss_name in self.loss_scale.keys():
            loss[loss_name] *= min(self.step * 0.01 + 1, self.loss_scale[loss_name])

        outputs = merge(output_list) if need_outputs else None

        return outputs, loss

    def save_model(self, path):
        torch.save(self.state_dict(), path)
