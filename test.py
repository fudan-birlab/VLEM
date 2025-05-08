import pickle
from model import EMModel
from utils import *

step_id = 5000

random_idx = 0

mode = 'distribute'
# mode = 'merge'

dataset_mode = 'EpiGibson'
# dataset_mode = 'patternDataset'
num_action, num_what, num_where, num_when = 100, 50, 20, 10
# num_action, num_what, num_where, num_when = 50, 20, 10, 5
# num_action, num_what, num_where, num_when = 20, 10, 5, 3

if dataset_mode == 'EpiGibson':
    exp_dir = f'outputs/{dataset_mode}-{mode}-{random_idx}/'
else:
    exp_dir = f'outputs/{dataset_mode}-{mode}-{num_action}-{num_what}-{num_where}-{num_when}-{random_idx}/'
ckpt_path = exp_dir + f'checkpoint-{step_id}.pt'
print(f'-----------  exp_dir : {exp_dir}  -----------\n')

attribute_list = ["whole-event"] if mode == 'merge' else ["where", "what", "when"]

model = EMModel(
    vision_input_size=1024,
    text_input_size=1024,
    # Event
    attribute_list=attribute_list,
    event_size=1024*(3 if mode == 'merge' else 1),
    # working memory
    wm_size=256,
    num_slots=7,
    # Entorihnal
    ento_size=1024,
    # episodic memory
    em_size=1024 * (3 if mode == 'merge' else 1),
    additional_iteration=2,
    drop=0.,
    # plan
    plan_size=1024 if dataset_mode == 'EpiGibson' else 3072,
    mode=mode,
).cuda()
model.load_state_dict(torch.load(ckpt_path))

model.eval()

test_data = torch.load(exp_dir + 'test_data.pt')
test_data = batch2cuda(test_data)

if __name__ == '__main__':
    eval_results = {}
    sigma_list =  [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    with torch.no_grad():
        print('--- original model evaluation ---')
        run_metrics_attractor(model, test_data)
        for sigma in sigma_list:
            print(f'noise level -- sigma = {sigma}')
            eval_results[sigma] = inference(model, test_data, sigma=sigma, exp_dir=exp_dir, filename_tag=f'EMmodel-{step_id}', save_outputs=sigma==0)
        
        with open(exp_dir + f'eval-results.pkl', 'wb') as f:
            pickle.dump(eval_results, f)

        if mode == 'merge':
            eval_results_hopfield = {}

            curEvent_target = test_data['curEvent_target']
            curEvent_target = torch.cat([curEvent_target[attribute] for attribute in model.attribute_list], -1)
            curEvent_target_unique = torch.unique(curEvent_target, dim=0)
            curEvent_target_unique = model.module_dict['episodic memory'].Tensor2dict(curEvent_target_unique)
            EM_target_dict = model.module_dict['events'].event2em(curEvent_target_unique)
            EM_target = model.eventDict2Tensor(EM_target_dict)
            model.to_hopfield(EM_target)

            print('--- hopfield model evaluation ---')
            for sigma in sigma_list:
                print(f'noise level -- sigma = {sigma}')
                eval_results_hopfield[sigma] = inference(model, test_data, sigma=sigma, exp_dir=exp_dir, filename_tag=f'hopfield-{step_id}', save_outputs=sigma==0)

            with open(exp_dir + f'eval-results-hopfield.pkl', 'wb') as f:
                pickle.dump(eval_results_hopfield, f)
