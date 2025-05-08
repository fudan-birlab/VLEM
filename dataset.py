import os
import torch
import decord
decord.bridge.set_bridge('torch')
import numpy as np
from einops import rearrange

import queue

class Tree():

    def __init__(self, num_node, edge=None):
        self.num_node = num_node
        if edge is None:
            self.edge = np.zeros((num_node, num_node), dtype=np.uint8)
            for i in range(1, num_node):
                link_to = np.random.randint(i)
                self.edge[link_to, i] = 1
                self.edge[i, link_to] = 1
        else:
            self.edge = edge

    def random_neibor(self, node_id):
        return np.random.choice(self.neibors(node_id))

    def neibors(self, node_id):
        return np.where(self.edge[node_id] > 0)[0]

    def find_path(self, node_st, node_ed):
        if node_st == node_ed:
            return [node_st]
        node_from = np.zeros(self.num_node, dtype=np.int32) - 1
        Q = queue.Queue()
        Q.put(node_st)
        while not Q.empty():
            node_id = Q.get()
            if node_id == node_ed:
                break
            for item in self.neibors(node_id):
                if node_from[item] == -1:
                    node_from[item] = node_id
                    Q.put(item)
        path = [node_ed]
        while path[-1] != node_st:
            path.append(node_from[path[-1]])
        return path[::-1]

class patternDataset(torch.utils.data.Dataset):

    def float2spike(self, latents, ratio=0.5):
        assert len(latents.shape) == 2
        s1 = latents >= torch.topk(latents, k=round(latents.shape[-1]*ratio), dim=-1, largest=True,  sorted=True).values[:,-1:]
        return s1 * 2. - 1.

    def __init__(
            self,
            path='./data/pattern/',
            num_action=40,
            num_what=20,
            num_where=10,
            num_when=4,
            input_size=1024,
            hidden_size=1024,
            sparseness=0.5,

            num_videos = 5,
            window_size = 256,
            phase='train',
            random_index=0,
        ):
        super().__init__()

        os.makedirs(path, exist_ok=True)
        self.path = path

        self.num_what = num_what
        self.num_where = num_where
        self.num_when = num_when
        self.num_action = num_action

        self.window_size = window_size
        self.window_slide = self.window_size // 2

        # ---------------------- meta data ---------------------
        meta_data_path = path + f'meta_data-{num_where}-{num_what}-{num_when}-{num_action}-{input_size}-{hidden_size}-{sparseness}-{random_index}.pt'
        if not os.path.exists(meta_data_path):
            meta_data = {
                # map
                "map_num_node" : self.num_where,
                "map_edge" : Tree(self.num_where).edge,
                # event-level
                "where_pattern" : (torch.rand(num_where, hidden_size) < sparseness) * 2. - 1.,
                "what_pattern"  : (torch.rand(num_what, hidden_size)  < sparseness) * 2. - 1.,
                "when_pattern"  : (torch.rand(num_when, hidden_size)  < sparseness) * 2. - 1.,
                "what_action_list" : [
                    np.random.randint(0, self.num_action, (np.random.randint(2,6)))
                    for i in range(num_what)
                ],
                # action-level
                "action_pattern" : (torch.rand(num_action, hidden_size) < sparseness) * 2. - 1.,
                "action_where" : [np.random.randint(self.num_where) for i in range(num_action)],
                # event2sensory weight
                "W1" : torch.randn(hidden_size*3, int(input_size ** 0.5)),
                "W2" : torch.randn(int(input_size ** 0.5), input_size*2),
            }
            torch.save(meta_data, meta_data_path)
        meta_data = torch.load(meta_data_path)
        self.traversal_map = Tree(meta_data['map_num_node'], meta_data['map_edge'])
        self.where_pattern = meta_data['where_pattern']
        self.what_pattern = meta_data['what_pattern']
        self.when_pattern = meta_data['when_pattern']
        self.what_action_list = meta_data['what_action_list']
        self.action_pattern = meta_data['action_pattern']
        self.action_where = meta_data['action_where']
        self.W1 = meta_data['W1']
        self.W2 = meta_data['W2']
        # ---------------------- meta data ---------------------

        # ---------------------- all data ---------------------
        all_data_path = path + f'all_data-{num_where}-{num_what}-{num_when}-{num_action}-{input_size}-{hidden_size}-{sparseness}-{phase}-{num_videos}-{random_index}.pt'
        if not os.path.exists(all_data_path):
            self.action_annot = []
            self.where_annot = []
            self.what_annot = []
            self.when_annot = []
            self.event_annot = []

            self.event_desc_feature = [] # what
            self.room_feature = [] # where
            self.time_feature = [] # when

            self.video_feature = []
            self.action_desc_feature = []

            self.next_event_idx = []
            self.next_room_idx = []
            self.next_time_idx = []
            self.next_mergedEvent_idx = []

            self.plan = []

            for video_id in range(num_videos):
                what_annot = []
                where_annot = []
                when_annot = []
                action_annot = []
                where_ing = self.action_where[self.what_action_list[0][0]]
                for what_id in range(num_what):
                    when_id = what_id * num_when // num_what
                    for action_id in self.what_action_list[what_id]:
                        path = self.traversal_map.find_path(where_ing, self.action_where[action_id])
                        for where_id in path:
                            for t in range(np.random.randint(20, 100)):
                                action_annot.append(action_id)
                                where_annot.append(where_id)
                                what_annot.append(what_id)
                                when_annot.append(when_id)
                        where_ing = path[-1]
                self.what_annot.append(np.array(what_annot))
                self.where_annot.append(np.array(where_annot))
                self.when_annot.append(np.array(when_annot))
                self.action_annot.append(np.array(action_annot))

                self.room_feature.append(self.where_pattern[where_annot])
                self.event_desc_feature.append(self.what_pattern[what_annot])
                self.time_feature.append(self.when_pattern[when_annot])

                event_pattern = torch.cat([self.room_feature[-1], self.event_desc_feature[-1], self.time_feature[-1]], -1)

                input_pattern = 2. * (((event_pattern @ self.W1) + torch.randn(event_pattern.shape[0], self.W1.shape[1]) * 0.2) > 0) - 1.
                input_pattern = 2. * (((input_pattern @ self.W2) + torch.randn(event_pattern.shape[0], self.W2.shape[1]) * 0.2) > 0) - 1.

                self.video_feature.append(input_pattern[:,:input_size])
                self.action_desc_feature.append(input_pattern[:,input_size:])

                next_event_idx = [-1]
                next_room_idx = [-1]
                next_time_idx = [-1]
                next_mergedEvent_idx = [-1]
                for i in range(len(what_annot) - 1, 0, -1): # for -> len-1 ... 1
                    next_event_idx.append(next_event_idx[-1] if what_annot[i]==what_annot[i-1] else i)
                    next_room_idx.append(next_room_idx[-1] if where_annot[i]==where_annot[i - 1] else i)
                    next_time_idx.append(next_time_idx[-1] if when_annot[i]==when_annot[i - 1] else i)
                    next_mergedEvent_idx.append(min(min(next_event_idx[-1], next_room_idx[-1]), next_time_idx[-1]))
                next_event_idx = np.array(next_event_idx[::-1])
                next_room_idx = np.array(next_room_idx[::-1])
                next_time_idx = np.array(next_time_idx[::-1])
                next_mergedEvent_idx = np.array(next_mergedEvent_idx[::-1])

                self.next_event_idx.append(next_event_idx)
                self.next_room_idx.append(next_room_idx)
                self.next_time_idx.append(next_time_idx)
                self.next_mergedEvent_idx.append(next_mergedEvent_idx)

                self.plan.append(event_pattern.mean(0))

            all_data = {
                "video_feat" : self.video_feature,
                "text_feat" : self.action_desc_feature,
                "action_annot" : self.action_annot,
                "where_feat" : self.room_feature,
                "what_feat" : self.event_desc_feature,
                "when_feat" : self.time_feature,
                "next_where_annot" : self.next_room_idx,
                "next_what_annot" : self.next_event_idx,
                "next_when_annot" : self.next_time_idx,
                "next_mergedEvent_annot" : self.next_mergedEvent_idx,
                "where_annot" : self.where_annot,
                "what_annot" : self.what_annot,
                "when_annot" : self.when_annot,
                "plan" : self.plan,
            }
            torch.save(all_data, all_data_path)
        all_data = torch.load(all_data_path)
        self.video_feature = all_data["video_feat"]
        self.action_desc_feature = all_data["text_feat"]
        self.action_annot = all_data["action_annot"]
        self.room_feature = all_data["where_feat"]
        self.event_desc_feature = all_data["what_feat"]
        self.time_feature = all_data["when_feat"]
        self.next_room_idx = all_data["next_where_annot"]
        self.next_event_idx = all_data["next_what_annot"]
        self.next_time_idx = all_data["next_when_annot"]
        self.next_mergedEvent_idx = all_data["next_mergedEvent_annot"]
        self.where_annot = all_data["where_annot"]
        self.what_annot = all_data["what_annot"]
        self.when_annot = all_data["when_annot"]
        self.plan = all_data["plan"]
        # ---------------------- all data ---------------------

        self.data_list = []
        if phase == 'train':
            for video_id in range(num_videos):
                video_length = len(self.video_feature[video_id])
                for timid in range((video_length - self.window_size) // self.window_slide + 1):
                    tim_st = timid * self.window_slide
                    tim_ed = tim_st + self.window_size
                    self.data_list.append((video_id, tim_st, tim_ed))
        else:
            for video_id in range(num_videos):
                self.data_list.append((video_id, 0, len(self.video_feature[video_id])))

        print(f'pattern Dataset loaded. (len={self.__len__()}, #time points in first run={len(self.video_feature[0])}, unique events:{np.unique(self.where_annot[0] * num_what + self.what_annot[0]).shape[0]})')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        video_id, tim_st, tim_ed = self.data_list[index]

        video_feat = self.video_feature[video_id][tim_st:tim_ed]
        text_feat = self.action_desc_feature[video_id][tim_st:tim_ed]
        action_annot = self.action_annot[video_id][tim_st:tim_ed]

        where_feat = self.room_feature[video_id][tim_st:tim_ed]
        what_feat = self.event_desc_feature[video_id][tim_st:tim_ed]
        when_feat = self.time_feature[video_id][tim_st:tim_ed]

        next_where_feat = self.room_feature[video_id][self.next_mergedEvent_idx[video_id][tim_st:tim_ed]]
        next_what_feat = self.event_desc_feature[video_id][self.next_mergedEvent_idx[video_id][tim_st:tim_ed]]
        next_when_feat = self.time_feature[video_id][self.next_mergedEvent_idx[video_id][tim_st:tim_ed]]

        where_annot = torch.from_numpy(self.where_annot[video_id][tim_st:tim_ed])
        what_annot = torch.from_numpy(self.what_annot[video_id][tim_st:tim_ed])
        when_annot = torch.from_numpy(self.when_annot[video_id][tim_st:tim_ed])

        return {
            'video_feat' : video_feat.float(),
            'text_feat' : text_feat.float(),
            'action_annot' : action_annot,
            'action_annot_text': ['' for i in range(tim_st, tim_ed)],
            'where_feat' : where_feat.float(),
            'what_feat' : what_feat.float(),
            'when_feat' : when_feat.float(),
            'next_where_feat' : next_where_feat.float(),
            'next_what_feat' : next_what_feat.float(),
            'next_when_feat' : next_when_feat.float(),
            'what_annot' : what_annot,
            'where_annot' : where_annot,
            'when_annot' : when_annot,
            'what_annot_text': ['' for i in range(tim_st, tim_ed)],
            'where_annot_text': ['' for i in range(tim_st, tim_ed)],
            'when_annot_text': ['' for i in range(tim_st, tim_ed)],
            'plan' : self.plan[video_id][None].repeat(tim_ed-tim_st, 1),
        }

class EpiMemDataset(torch.utils.data.Dataset):

    def preprocess(self):
        import open_clip
        from torchvision import transforms

        model, _, self.preprocessor = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
        self.tokenizer = open_clip.get_tokenizer('ViT-H-14')
        model.to('cuda')

        for video_id in range(self.num_videos):
            room_feature_path = os.path.join(self.path, '%s/%04d/room_feature.pt' % (self.phase, video_id + 1))
            action_desc_feature_path = os.path.join(self.path, '%s/%04d/action_desc_feature.pt' % (self.phase, video_id + 1))
            event_desc_feature_path = os.path.join(self.path, '%s/%04d/event_desc_feature.pt' % (self.phase, video_id + 1))
            video_feature_path = os.path.join(self.path, '%s/%04d/video_feature.pt' % (self.phase, video_id + 1))

            if not os.path.exists(os.path.join(self.path, '%s/%04d/t2pattern.pt' % (self.phase, video_id + 1))):
                t2pattern = torch.randn(3, 1024)
                torch.save(
                    t2pattern,
                    os.path.join(self.path, '%s/%04d/t2pattern.pt' % (self.phase, video_id + 1))
                )

            if not os.path.exists(os.path.join(self.path, '%s/%04d/plan.pt' % (self.phase, video_id + 1))):
                plan_emb = torch.randn(1024)
                torch.save(
                    plan_emb.flatten().cpu(),
                    os.path.join(self.path, '%s/%04d/plan.pt' % (self.phase, video_id + 1))
                )

            if not (os.path.exists(action_desc_feature_path) and os.path.exists(event_desc_feature_path)):
                print(f'processing description ... {video_id}/{self.num_videos}')
                action_desc_feature = []
                event_desc_feature = []
                room_feature = []
                action_desc_list = np.load(os.path.join(self.path, '%s/%04d/action_desc_list.npy') % (self.phase, video_id + 1))
                event_desc_list = np.load(os.path.join(self.path, '%s/%04d/event_desc_list.npy') % (self.phase, video_id + 1))
                room_list = np.load(os.path.join(self.path, '%s/%04d/room_list.npy') % (self.phase, video_id + 1))
                action_desc_list[action_desc_list==""] = "idle"
                event_desc_list[event_desc_list==""] = "idle"
                for timid in range(len(action_desc_list)):
                    if timid % 10 == 0:
                        print(f'{timid} / {len(action_desc_list)}')
                    action_desc = action_desc_list[timid]
                    event_desc = event_desc_list[timid]
                    room = room_list[timid]
                    
                    action_token = self.tokenizer(action_desc).to('cuda')
                    event_token = self.tokenizer(event_desc).to('cuda')
                    room_token = self.tokenizer(room).to('cuda')

                    with torch.no_grad(), torch.cuda.amp.autocast():
                        action_emb = model.encode_text(action_token)
                        event_emb = model.encode_text(event_token)
                        room_emb = model.encode_text(room_token)
                        action_emb /= action_emb.norm(dim=-1, keepdim=True)
                        event_emb /= event_emb.norm(dim=-1, keepdim=True)
                        room_emb /= room_emb.norm(dim=-1, keepdim=True)

                    action_desc_feature.append(action_emb.flatten().cpu()[None])
                    event_desc_feature.append(event_emb.flatten().cpu()[None])
                    room_feature.append(room_emb.flatten().cpu()[None])

                action_desc_feature = torch.cat(action_desc_feature, 0)
                event_desc_feature = torch.cat(event_desc_feature, 0)
                room_feature = torch.cat(room_feature, 0)
                torch.save(action_desc_feature, action_desc_feature_path)
                torch.save(event_desc_feature, event_desc_feature_path)
                torch.save(room_feature, room_feature_path)

            if not os.path.exists(video_feature_path):
                print(f'processing visual input ... {video_id}/{self.num_videos}')

                video_path = os.path.join(self.path, '%s/%04d/visual_input.mp4' % (self.phase, video_id + 1))
                video = decord.VideoReader(video_path, width=self.width, height=self.height)
                video_length = len(video)
                sample_index = [timid * self.sample_frame_rate for timid in range(video_length)]

                video_feature = []
                for timid in range(len(sample_index)):
                    if timid % 10 == 0:
                        print(f'{timid} / {len(sample_index)}')
                    video_data = video.get_batch([timid * self.sample_frame_rate])
                    video_data = rearrange(video_data, "f h w c -> f c h w") / 255.
                    pixel_values = video_data[0].cuda()
                    img = self.preprocessor(transforms.ToPILImage()(pixel_values)).unsqueeze(0).to(pixel_values.device, dtype=torch.float16)
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        image_features = model.encode_image(img)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        video_feature.append(image_features.flatten().cpu()[None])
                video_feature = torch.cat(video_feature, 0)
                torch.save(video_feature, video_feature_path)
        del model
        torch.cuda.empty_cache()

    def __init__(self,
                 path='./data/EpiGibson',

                 width: int = 1024,
                 height: int = 1024,
                 sample_frame_rate: int = 1,
                 n_sample_frames: int = 1,

                 num_videos = 1,
                 window_size = 2048,

                 phase='train',
        ):

        super().__init__()

        self.width = width
        self.height = height
        self.sample_frame_rate = sample_frame_rate
        self.n_sample_frames = n_sample_frames

        self.num_videos = num_videos
        self.window_size = window_size
        self.window_slide = window_size // 2
        self.phase = phase
        self.path = path

        self.t2time = {0:'morning', 1:'afternoon', 2:'night'}
        
        if np.any(
            [not os.path.exists(os.path.join(self.path, '%s/%04d/t2pattern.pt' % (self.phase, video_id + 1))) for video_id in range(self.num_videos)] + 
            [not os.path.exists(os.path.join(self.path, '%s/%04d/plan.pt' % (self.phase, video_id + 1))) for video_id in range(self.num_videos)] + 
            [not os.path.exists(os.path.join(path, '%s/%04d/video_feature.pt' % (phase, video_id + 1))) for video_id in range(self.num_videos)] + 
            [not os.path.exists(os.path.join(path, '%s/%04d/event_desc_feature.pt' % (phase, video_id + 1))) for video_id in range(self.num_videos)]
        ):
            self.preprocess()

        self.action_annot = []
        self.event_annot = [] # what
        self.room_annot = [] # where
        self.time_annot = []

        self.video_feature = []
        self.action_desc_feature = []

        self.event_desc_feature = []
        self.room_feature = []
        self.time_feature = []

        self.action_annot_text = []
        self.event_annot_text = []
        self.room_annot_text = []
        self.time_annot_text = []

        self.next_event_idx = []
        self.next_room_idx = []
        self.next_time_idx = []

        self.plan = []

        for video_id in range(self.num_videos):

            self.plan.append(torch.load(os.path.join(self.path, '%s/%04d/plan.pt') % (self.phase, video_id + 1)))
            t2pattern = torch.load(os.path.join(self.path, '%s/%04d/t2pattern.pt') % (self.phase, video_id + 1))

            # annot
            action_desc_list = np.load(os.path.join(self.path, '%s/%04d/action_desc_list.npy') % (self.phase, video_id + 1))
            event_desc_list = np.load(os.path.join(self.path, '%s/%04d/event_desc_list.npy') % (self.phase, video_id + 1))
            room_list = np.load(os.path.join(self.path, '%s/%04d/room_list.npy') % (self.phase, video_id + 1))

            action_dict = {item:idx for idx, item in enumerate(np.unique(action_desc_list))}
            action_annot = np.array([action_dict[item] for item in action_desc_list])
            event_dict = {item:idx for idx, item in enumerate(np.unique(event_desc_list))}
            event_annot = np.array([event_dict[item] for item in event_desc_list])
            room_dict = {item:idx for idx, item in enumerate(np.unique(room_list))}
            room_annot = np.array([room_dict[item] for item in room_list])
            
            time_annot = np.zeros_like(room_annot, dtype=np.int32)
            time_annot[room_annot.shape[0]//3:room_annot.shape[0]//3*2] = 1
            time_annot[room_annot.shape[0]//3*2:] = 2

            next_event_idx = [-1]
            next_room_idx = [-1]
            next_time_idx = [-1]
            for i in range(len(event_annot) - 1, 0, -1): # for -> len-1 ... 1
                next_event_idx.append(next_event_idx[-1] if event_annot[i]==event_annot[i-1] else i)
                next_room_idx.append(next_room_idx[-1] if room_annot[i]==room_annot[i - 1] else i)
                next_time_idx.append(next_time_idx[-1] if time_annot[i]==time_annot[i - 1] else i)
            next_event_idx = np.array(next_event_idx[::-1])
            next_room_idx = np.array(next_room_idx[::-1])
            next_time_idx = np.array(next_time_idx[::-1])

            self.action_annot.append(action_annot)
            self.event_annot.append(event_annot)
            self.room_annot.append(room_annot)
            self.time_annot.append(time_annot)

            self.action_annot_text.append(action_desc_list)
            self.event_annot_text.append(event_desc_list)
            self.room_annot_text.append(room_list)

            self.time_annot_text.append([self.t2time[t] for t in time_annot])
            time_feature = t2pattern[time_annot]

            self.next_event_idx.append(next_event_idx)
            self.next_room_idx.append(next_room_idx)
            self.next_time_idx.append(next_time_idx)

            action_desc_feature_path = os.path.join(path, '%s/%04d/action_desc_feature.pt' % (phase, video_id + 1))
            event_desc_feature_path = os.path.join(path, '%s/%04d/event_desc_feature.pt' % (phase, video_id + 1))
            room_feature_path = os.path.join(path, '%s/%04d/room_feature.pt' % (phase, video_id + 1))
            video_feature_path = os.path.join(path, '%s/%04d/video_feature.pt' % (phase, video_id + 1))

            self.video_feature.append(torch.load(video_feature_path))
            self.action_desc_feature.append(torch.load(action_desc_feature_path))

            self.event_desc_feature.append(torch.load(event_desc_feature_path))
            self.room_feature.append(torch.load(room_feature_path))
            self.time_feature.append(time_feature)

        print(f'EpiMem dataset ({self.phase}) load finished. (window_size={self.window_size})')

        self.data_list = []
        if phase == 'train':
            for video_id in range(self.num_videos):
                video_length = len(self.video_feature[video_id])
                for timid in range((video_length - self.window_size) // self.window_slide + 1):
                    tim_st = timid * self.window_slide
                    tim_ed = tim_st + self.window_size
                    self.data_list.append((video_id, tim_st, tim_ed))
        else:
            for video_id in range(self.num_videos):
                self.data_list.append((video_id, 0, len(self.video_feature[video_id])))

    def __len__(self):
        return len(self.data_list)

    def float2spike(self, latents, ratio=0.5):
        assert len(latents.shape) == 2
        s1 = latents >= torch.topk(latents, k=round(latents.shape[-1]*ratio), dim=-1, largest=True,  sorted=True).values[:,-1:]
        return s1 * 2. - 1.

    def __getitem__(self, index):
        video_id, tim_st, tim_ed = self.data_list[index]

        video_feat = self.video_feature[video_id][tim_st:tim_ed]
        text_feat = self.action_desc_feature[video_id][tim_st:tim_ed]

        action_annot = self.action_annot[video_id][tim_st:tim_ed]
        action_annot_text = [[item] for item in self.action_annot_text[video_id][tim_st:tim_ed]]

        where_feat = self.room_feature[video_id][tim_st:tim_ed]
        what_feat = self.event_desc_feature[video_id][tim_st:tim_ed]
        when_feat = self.time_feature[video_id][tim_st:tim_ed]

        next_where_feat = self.room_feature[video_id][self.next_room_idx[video_id][tim_st:tim_ed]]
        next_what_feat = self.event_desc_feature[video_id][self.next_event_idx[video_id][tim_st:tim_ed]]
        next_when_feat = self.time_feature[video_id][self.next_time_idx[video_id][tim_st:tim_ed]]

        where_annot = self.room_annot[video_id][tim_st:tim_ed]
        where_annot_text = [[item] for item in self.room_annot_text[video_id][tim_st:tim_ed]]
        what_annot = self.event_annot[video_id][tim_st:tim_ed]
        what_annot_text = [[item] for item in self.event_annot_text[video_id][tim_st:tim_ed]]
        when_annot = self.time_annot[video_id][tim_st:tim_ed]
        when_annot_text = [[item] for item in self.time_annot_text[video_id][tim_st:tim_ed]]

        return {
            'video_feat' : self.float2spike(video_feat.float()),
            'text_feat' : self.float2spike(text_feat.float()),
            'action_annot' : action_annot,
            'action_annot_text': action_annot_text,
            'where_feat' : self.float2spike(where_feat.float()),
            'what_feat' : self.float2spike(what_feat.float()),
            'when_feat' : when_feat.float(),
            'next_where_feat' : self.float2spike(next_where_feat.float()),
            'next_what_feat' : self.float2spike(next_what_feat.float()),
            'next_when_feat' : next_when_feat.float(),
            'what_annot' : what_annot,
            'where_annot' : where_annot,
            'when_annot' : when_annot,
            'what_annot_text' : what_annot_text,
            'where_annot_text' : where_annot_text,
            'when_annot_text': when_annot_text,
            'plan' : self.plan[video_id][None].repeat(tim_ed-tim_st, 1),
        }
