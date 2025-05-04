import logging
import pickle
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, Dataset
__all__ = ['MMDataLoader']
logger = logging.getLogger('MMSA')

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATASET_MAP = {
            'first_impression': self.__init_first_impression
        }
        DATASET_MAP[args['dataset_name']]()

    def __init_first_impression(self):

        with open("dataset/first_impression/" + str(self.mode) + "/annotation_" + str(self.mode) +".pkl", 'rb') as f:
            label_data = pickle.load(f, encoding='latin1')
            self.labels = {
                'agreeableness': [],
                'openness': [],
                'neuroticism': [],
                'extraversion': [],
                'conscientiousness': []
            }


        self.ids = []
        audio_folder_path = "dataset/first_impression/" + str(self.mode) + '/audio/'
        video_folder_path = "dataset/first_impression/" + str(self.mode) + '/video/'
        text_folder_path = "dataset/first_impression/" + str(self.mode) + '/text/'

        empty_indices = []
        audio_arrays = []
        i = 0
        for file in sorted(os.listdir(audio_folder_path)):  # optional sort for consistent order
            if file.endswith('.npy'):
                file_path = os.path.join(audio_folder_path, file)
                data = np.load(file_path, allow_pickle=True)
                if data.size == 0:
                    empty_indices.append(i)
                    i += 1
                    continue
                if data.shape[0]!=5:
                    empty_indices.append(i)
                    i += 1
                    continue
                data = np.mean(data, axis=1)
                audio_arrays.append(data)

                file_to_check = file[:-4]
                self.labels['agreeableness'].append(label_data['agreeableness'][file_to_check + ".mp4"])
                self.labels['openness'].append(label_data['openness'][file_to_check + ".mp4"])
                self.labels['neuroticism'].append(label_data['neuroticism'][file_to_check + ".mp4"])
                self.labels['extraversion'].append(label_data['extraversion'][file_to_check + ".mp4"])
                self.labels['conscientiousness'].append(label_data['conscientiousness'][file_to_check + ".mp4"])

                i+=1

        # Combine into one array (vertically stacked, if shapes match)
        self.audio = np.stack(audio_arrays, axis=0)

        i=0
        video_arrays = []
        for file in sorted(os.listdir(video_folder_path)):  # optional sort for consistent order
            if file.endswith('.npy'):
                file_path = os.path.join(video_folder_path, file)
                data = np.load(file_path, allow_pickle=True)
                if data.size == 0:
                    empty_indices.append(i)
                    i += 1
                    continue
                if i in empty_indices:
                    i += 1
                    continue
                data = np.mean(data, axis=1)
                video_arrays.append(data)
                i+=1
        # Combine into one array (vertically stacked, if shapes match)
        self.vision = np.stack(video_arrays, axis=0)

        i=0
        text_arrays = []
        for file in sorted(os.listdir(text_folder_path)):  # optional sort for consistent order
            if file.endswith('.npy'):
                file_path = os.path.join(text_folder_path, file)
                data = np.load(file_path, allow_pickle=True)
                if data.size == 0:
                    empty_indices.append(i)
                    i += 1
                    continue
                if i in empty_indices:
                    i += 1
                    continue
                text_arrays.append(data)
                self.ids.append(file)
                i+=1

        # Combine into one array (vertically stacked, if shapes match)
        self.text = np.stack(text_arrays, axis=0)

        self.raw_text = self.text

        self.labels['agreeableness'] = np.array(self.labels['agreeableness']).astype(np.float32)
        self.labels['openness'] = np.array(self.labels['openness']).astype(np.float32)
        self.labels['neuroticism'] = np.array(self.labels['neuroticism']).astype(np.float32)
        self.labels['extraversion'] = np.array(self.labels['extraversion']).astype(np.float32)
        self.labels['conscientiousness'] = np.array(self.labels['conscientiousness']).astype(np.float32)

        logger.info(f"{self.mode} samples: {self.labels['openness'].shape}")

    def __normalize(self):
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels['openness'])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'raw_text': self.raw_text[index],
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        } 
        if not self.args['need_data_aligned']:
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]
        return sample

def MMDataLoader(args, num_workers):

    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args['seq_lens'] = datasets['train'].get_seq_len() 

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args['batch_size'],
                       num_workers=num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }
    
    return dataLoader


class DownstreamDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        self.__init_downstream_data()


    def __init_downstream_data(self):

        with open("reconstructed_outputs_" + str(self.mode) + ".pkl", 'rb') as f:
            loaded_dataset = pickle.load(f)

        self.labels = {
            'agreeableness': [],
            'openness': [],
            'neuroticism': [],
            'extraversion': [],
            'conscientiousness': []
        }

        self.original_text = []
        self.original_vision = []
        self.original_audio = []
        self.ground_truth_vision = []
        self.ground_truth_audio = []
        self.ground_truth_text = []
        self.generated_vision = []
        self.generated_audio = []

        for data in loaded_dataset:
            self.labels['agreeableness'].append([data['agreeableness']])
            self.labels['openness'].append([data['openness']])
            self.labels['neuroticism'].append([data['neuroticism']])
            self.labels['extraversion'].append([data['extraversion']])
            self.labels['conscientiousness'].append([data['conscientiousness']])
            self.original_text.append(data['original_text'])
            self.original_vision.append(data['original_vision'])
            self.original_audio.append(data['original_audio'])
            self.ground_truth_vision.append(data['ground_truth_vision'])
            self.ground_truth_audio.append(data['ground_truth_audio'])
            self.ground_truth_text.append(data['ground_truth_text'])
            self.generated_vision.append(data['generated_vision'])
            self.generated_audio.append(data['generated_audio'])

        logger.info(f"Downstream Dataset {self.mode} samples: {len(self.labels['openness'])}")


    def __len__(self):
        return len(self.labels['openness'])

    def __getitem__(self, index):
        sample = {
            'original_text': torch.Tensor(self.original_text[index]),
            'original_vision': torch.Tensor(self.original_vision[index]),
            'original_audio': torch.Tensor(self.original_audio[index]),
            'ground_truth_text': torch.Tensor(self.ground_truth_text[index]),
            'ground_truth_vision': torch.Tensor(self.ground_truth_vision[index]),
            'ground_truth_audio': torch.Tensor(self.ground_truth_audio[index]),
            'generated_vision': torch.Tensor(self.generated_vision[index]),
            'generated_audio': torch.Tensor(self.generated_audio[index]),
            'index': index,
            'labels': {k: torch.Tensor(v[index]) for k, v in self.labels.items()}
        }
        return sample


def DownstreamDataLoader(args, num_workers):
    datasets = {
        'train': DownstreamDataset(args, mode='train'),
        'valid': DownstreamDataset(args, mode='valid'),
        'test': DownstreamDataset(args, mode='test')
    }

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args['batch_size'],
                       num_workers=num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }

    return dataLoader