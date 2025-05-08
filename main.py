import os
import torch
from dataset import EpiMemDataset, patternDataset
from model import EMModel, traverse_apply, to_cpu

import numpy as np
import cv2

from utils import *

if __name__ == '__main__':

    mode = 'distribute' # 'where' + 'what' + 'when'
    # mode = 'merge' # concat ['where' + 'what' + 'when'] -> ['whole-event']

    dataset_mode = 'EpiGibson'
    # dataset_mode = 'patternDataset'
    num_action, num_what, num_where, num_when = 100, 50, 20, 10
    # num_action, num_what, num_where, num_when = 50, 20, 10, 5
    # num_action, num_what, num_where, num_when = 20, 10, 5, 3

    random_index = 0
    training_window_size = 256
    batch_size = 256 * 256 // training_window_size

    loss_scale = {
        'loss episodic memory target' : 5,
        'loss episodic memory attractor' : 5,
        'loss current episodic memory' : 5,
        'loss contrastive episodic memory' : 5,
    }
    if dataset_mode == 'EpiGibson':
        exp_dir = f'outputs/{dataset_mode}-{mode}-{random_index}/'
    else:
        exp_dir = f'outputs/{dataset_mode}-{mode}-{num_action}-{num_what}-{num_where}-{num_when}-{random_index}/'
    print(f'-----------  exp_dir : {exp_dir}  -----------\n')
    print(f'-----------    mode  : {mode}  -----------\n')
    attribute_list = ["where", "what", "when"]

    model = EMModel(
        vision_input_size=1024,
        text_input_size=1024,
        # Event
        attribute_list=["whole-event"] if mode == 'merge' else attribute_list,
        event_size=1024 * (len(attribute_list) if mode == 'merge' else 1),
        # working memory
        wm_size=256,
        num_slots=7,
        # Entorihnal
        ento_size=1024,
        # episodic memory
        em_size=1024 * (len(attribute_list) if mode == 'merge' else 1),
        additional_iteration=2,
        drop=0.,
        # plan
        plan_size=1024 if dataset_mode == 'EpiGibson' else 3072,
        mode=mode,
        loss_scale=loss_scale,
    ).cuda()

    if dataset_mode == 'EpiGibson':
        dataset = EpiMemDataset(phase='train', window_size=training_window_size, num_videos=2)
        test_dataset = EpiMemDataset(phase='test')
    else:
        print(f'pattern dataset\n  num_action : {num_action}\n  num_what : {num_what}\n  num_where : {num_where}\n  num_when : {num_when}\n  training window size : {training_window_size}\n\n')

        dataset = patternDataset(
            phase='train',
            window_size=training_window_size,
            num_action=num_action,
            num_what=num_what,
            num_where=num_where,
            num_when=num_when,
            random_index=random_index,
        )
        test_dataset = patternDataset(
            phase='test',
            num_videos=1,
            num_action=num_action,
            num_what=num_what,
            num_where=num_where,
            num_when=num_when,
            random_index=random_index,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, dataset.__len__()),
        shuffle=True,
        num_workers=min(8, dataset.__len__()),
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    lr = 2e-4
    decay_half_per_steps = 500
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5 ** (100./decay_half_per_steps))

    print(f'optimizer -- lr : {lr}; decay_half_per_steps : {decay_half_per_steps}\n')

    step_id = 0
    max_training_steps = 5000
    os.makedirs(exp_dir, exist_ok=True)
    test_data_path = exp_dir + 'test_data.pt'
    while True:
        for idx, batch in enumerate(dataloader):
            model.train()
            step_id += 1

            enable_test = step_id % 500 == 0

            vision_sensory_input, text_sensory_input, curEvent, nxtEvent, plan, annot, annot_where, annot_when, annot_action = batch_prepare(batch, mode=mode, attribute_list=attribute_list)

            outputs, loss = model(
                vision_sensory_input=vision_sensory_input,
                text_sensory_input=text_sensory_input,
                curEvent=curEvent,
                nxtEvent=nxtEvent,
                plan=plan,
                need_outputs=False
            )
            outputs = outputs_prepare(outputs, mode, attribute_list)

            loss_sum, loss_info = loss_process(loss)
            print(f'\ntraining step -- {step_id}:\n' + loss_info)

            loss_sum.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            model.constrain_attractor()

            print(f'Learning Rate = {scheduler.get_last_lr()}')

            if enable_test:
                model.eval()
                model.save_model(path=exp_dir+f'checkpoint-{step_id}.pt')
                torch.cuda.empty_cache()
                with torch.no_grad():
                    for test_idx, test_batch in enumerate(test_dataloader):
                        vision_sensory_input, text_sensory_input, curEvent, nxtEvent, plan, annot, annot_where, annot_when, annot_action = batch_prepare(test_batch, mode, attribute_list)

                        if not os.path.exists(test_data_path):
                            np.save(exp_dir + 'annot_text.npy', test_batch['what_annot_text'])
                            np.save(exp_dir + 'annot_when_text.npy', test_batch['when_annot_text'])
                            np.save(exp_dir + 'annot_where_text.npy', test_batch['where_annot_text'])
                            np.save(exp_dir + 'annot_action_text.npy', test_batch['action_annot_text'])
                            test_data = {
                                'annot' : annot,
                                'annot_when' : annot_when,
                                'annot_where' : annot_where,
                                'annot_action' : annot_action,
                                'vision_sensory_input' : vision_sensory_input.cpu(),
                                'text_sensory_input' : text_sensory_input.cpu(),
                                'sensory_input' : torch.cat([vision_sensory_input, text_sensory_input], -1).cpu(),
                                'curEvent_target' : traverse_apply(curEvent, to_cpu),
                                'nxtEvent_target' : traverse_apply(nxtEvent, to_cpu),
                                'plan' : plan.cpu()
                            }
                            torch.save(test_data, test_data_path)

                        outputs, loss = model(
                            vision_sensory_input=vision_sensory_input,
                            text_sensory_input=text_sensory_input,
                            curEvent=curEvent,
                            nxtEvent=nxtEvent,
                            plan=plan,
                            need_outputs=True
                        )
                        outputs = outputs_prepare(outputs, mode, attribute_list)

                        test_data = torch.load(test_data_path)
                        test_data = batch2cuda(test_data)
                        run_metrics(model, test_data, outputs)
                        run_metrics_attractor(model, test_data)

                        loss_sum, loss_info = loss_process(loss)
                        print(f'\n  ---------------  test  ---------------  :\n' + loss_info)

                        cur_event = eventDict2Tensor(outputs['event prediction'], attribute_list)[:,:1]
                        wm = outputs['working memory'][:,:1]
                        em = eventDict2Tensor(outputs['episodic memory'], attribute_list)[:,:1]
                        annot = annot[:1]
                        annot_action = annot_action[:1]

                        interval = len(test_dataset.video_feature[0]) // 1000 # ~ window_size / 1000

                        cur_event = cur_event[::interval].detach().cpu()
                        em = em[::interval].detach().cpu()
                        annot = annot[:,::interval].detach().cpu()
                        wm = wm[::10*interval].detach().cpu()
                        annot_action = annot_action[:,::10*interval].detach().cpu()

                        img_all, img_list = heatMapPlot(cur_event, annot)
                        cv2.imwrite(exp_dir + f'training-event-{step_id}.png', img_all)
                        img_all, img_list = heatMapPlot(em, annot)
                        cv2.imwrite(exp_dir + f'training-em-{step_id}.png', img_all)

                        wm = torch.cat([torch.zeros_like(wm[:,:,:1], device=wm.device), wm], 2)
                        wm = wm.flatten(0, 2)[:,None]
                        img_all, img_list = heatMapPlot(wm, annot_action.flatten()[:,None].repeat(1,model.num_slots+1).reshape(1, -1))

                        cv2.imwrite(exp_dir + f'training-wm-{step_id}.png', img_all)
            if step_id >= max_training_steps:
                break
        if step_id >= max_training_steps:
            break

