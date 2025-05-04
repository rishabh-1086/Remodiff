import logging
import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str
logger = logging.getLogger('ReMoDiff')

class ReMoDiff():
    def __init__(self, args):
        self.args = args
        self.criterion_agreeableness = nn.MSELoss()
        self.criterion_openness = nn.MSELoss()
        self.criterion_neuroticism = nn.MSELoss()
        self.criterion_extraversion = nn.MSELoss()
        self.criterion_conscientiousness = nn.MSELoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def do_train(self, model, dataloader, return_epoch_results=False):
        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)
        # initilize results
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        '''
        # load pretrained
        origin_model = torch.load('pt/pretrained-{}.pth'.format(self.args.dataset_name))
        net_dict = model.state_dict()
        new_state_dict = {}
        for k, v in origin_model.items():
            k = k.replace('Model.', '')
            new_state_dict[k] = v
        net_dict.update(new_state_dict)
        model.load_state_dict(net_dict)
        '''

        while True:
            epochs += 1
            # train
            y_pred_agreeableness = []
            y_pred_openness = []
            y_pred_neuroticism = []
            y_pred_extraversion = []
            y_pred_conscientiousness = []

            y_true_agreeableness = []
            y_true_openness = []
            y_true_neuroticism = []
            y_true_extraversion = []
            y_true_conscientiousness = []

            losses = []
            model.train()
            train_loss = 0.0
            miss_one, miss_two = 0, 0  # num of missing one modal and missing two modal
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels_agreeableness = batch_data['labels']['agreeableness'].to(self.args.device)#.view(-1, 1)
                    labels_openness = batch_data['labels']['openness'].to(self.args.device)#.view(-1, 1)
                    labels_neuroticism = batch_data['labels']['neuroticism'].to(self.args.device)#.view(-1, 1)
                    labels_extraversion = batch_data['labels']['extraversion'].to(self.args.device)#.view(-1, 1)
                    labels_conscientiousness = batch_data['labels']['conscientiousness'].to(self.args.device)#.view(-1, 1)


                    outputs = model(text, audio, vision, missing='audio_and_vision')


                    # compute loss

                    task_loss_agreeableness = self.criterion_agreeableness(outputs['agreeableness'].view(-1, 1), labels_agreeableness)
                    task_loss_openness = self.criterion_openness(outputs['openness'].view(-1, 1), labels_openness)
                    task_loss_neuroticism = self.criterion_neuroticism(outputs['neuroticism'].view(-1, 1), labels_neuroticism)
                    task_loss_extraversion = self.criterion_extraversion(outputs['extraversion'].view(-1, 1), labels_extraversion)
                    task_loss_conscientiousness = self.criterion_conscientiousness(outputs['conscientiousness'].view(-1, 1), labels_conscientiousness)


                    loss_score_l = outputs['loss_score_l']
                    loss_score_v = outputs['loss_score_v']
                    loss_score_a = outputs['loss_score_a']
                    loss_rec = outputs['loss_rec']
                    combine_loss = task_loss_agreeableness + task_loss_openness + task_loss_neuroticism + task_loss_extraversion + task_loss_conscientiousness + 0.1 * (loss_score_l + loss_score_v + loss_score_a + loss_rec)

                    # backward
                    combine_loss.backward()
                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad],
                                                  self.args.grad_clip)
                    # store results
                    train_loss += combine_loss.item()

                    y_pred_agreeableness.append(outputs['agreeableness'].cpu())
                    y_pred_openness.append(outputs['openness'].cpu())
                    y_pred_neuroticism.append(outputs['neuroticism'].cpu())
                    y_pred_extraversion.append(outputs['extraversion'].cpu())
                    y_pred_conscientiousness.append(outputs['conscientiousness'].cpu())

                    y_true_agreeableness.append(labels_agreeableness.cpu())
                    y_true_openness.append(labels_openness.cpu())
                    y_true_neuroticism.append(labels_neuroticism.cpu())
                    y_true_extraversion.append(labels_extraversion.cpu())
                    y_true_conscientiousness.append(labels_conscientiousness.cpu())

                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()
            train_loss = train_loss / len(dataloader['train'])

            pred_agreeableness, true_agreeableness = torch.cat(y_pred_agreeableness), torch.cat(y_true_agreeableness)
            pred_openness, true_openness = torch.cat(y_pred_openness), torch.cat(y_true_openness)
            pred_neuroticism, true_neuroticism = torch.cat(y_pred_neuroticism), torch.cat(y_true_neuroticism)
            pred_extraversion, true_extraversion = torch.cat(y_pred_extraversion), torch.cat(y_true_extraversion)
            pred_conscientiousness, true_conscientiousness = torch.cat(y_pred_conscientiousness), torch.cat(y_true_conscientiousness)


            train_results_agreeableness = self.metrics(pred_agreeableness, true_agreeableness)
            train_results_openness = self.metrics(pred_openness, true_openness)
            train_results_neuroticism = self.metrics(pred_neuroticism, true_neuroticism)
            train_results_extraversion = self.metrics(pred_extraversion, true_extraversion)
            train_results_conscientiousness = self.metrics(pred_conscientiousness, true_conscientiousness)

            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> loss: {round(train_loss, 4)} "
                f">> Agreeableness: {dict_to_str(train_results_agreeableness)}"
                f">> Openness: {dict_to_str(train_results_openness)}"
                f">> Neuroticism: {dict_to_str(train_results_neuroticism)}"
                f">> Extraversion: {dict_to_str(train_results_extraversion)}"
                f">> Conscientiousness: {dict_to_str(train_results_conscientiousness)}"
            )
            print(f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
            f">> loss: {round(train_loss, 4)} "
            f">> Agreeableness: {dict_to_str(train_results_agreeableness)}"
            f">> Openness: {dict_to_str(train_results_openness)}"
            f">> Neuroticism: {dict_to_str(train_results_neuroticism)}"
            f">> Extraversion: {dict_to_str(train_results_extraversion)}"
            f">> Conscientiousness: {dict_to_str(train_results_conscientiousness)}")

            # validation
            val_results_agreeableness, val_results_openness, val_results_neuroticism, val_results_extraversion, val_results_conscientiousness = self.do_test(model, dataloader['valid'], mode="VAL")
            test_results_agreeableness, test_results_openness, test_results_neuroticism, test_results_extraversion, test_results_conscientiousness = self.do_test(model, dataloader['test'], mode="TEST")
            cur_valid = val_results_agreeableness[self.args.KeyEval]
            scheduler.step(val_results_agreeableness['Loss'])
            # save each epoch model
            #model_save_path = 'pt/' + str(epochs) + '.pth'  #Commented by Rishabh to save on space
            #torch.save(model.state_dict(), model_save_path)  #Commented by Rishabh to save on space

            print('Saving the model')
            torch.save(model.cpu().state_dict(), self.args.model_save_path)
            print('Model saved')
            model.to(self.args.device)

            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            # epoch results
            if return_epoch_results:
                train_results_agreeableness["Loss"] = train_loss
                train_results_openness["Loss"] = train_loss
                train_results_neuroticism["Loss"] = train_loss
                train_results_extraversion["Loss"] = train_loss
                train_results_conscientiousness["Loss"] = train_loss
                '''
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
                '''
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return #epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()

        y_pred_agreeableness = []
        y_pred_openness = []
        y_pred_neuroticism = []
        y_pred_extraversion = []
        y_pred_conscientiousness = []

        y_true_agreeableness = []
        y_true_openness = []
        y_true_neuroticism = []
        y_true_extraversion = []
        y_true_conscientiousness = []

        miss_one, miss_two = 0, 0

        eval_loss = 0.0
        if return_sample_results:
            ids = []
            sample_results_agreeableness = []
            sample_results_openness = []
            sample_results_neuroticism = []
            sample_results_extraversion = []
            sample_results_conscientiousness = []

            all_labels_agreeableness = []
            all_labels_openness = []
            all_labels_neuroticism = []
            all_labels_extraversion = []
            all_labels_conscientiousness = []

            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels_agreeableness = batch_data['labels']['agreeableness'].to(self.args.device)#.view(-1, 1)
                    labels_openness = batch_data['labels']['openness'].to(self.args.device)#.view(-1, 1)
                    labels_neuroticism = batch_data['labels']['neuroticism'].to(self.args.device)#.view(-1, 1)
                    labels_extraversion = batch_data['labels']['extraversion'].to(self.args.device)#.view(-1, 1)
                    labels_conscientiousness = batch_data['labels']['conscientiousness'].to(self.args.device)#.view(-1, 1)

                    outputs = model(text, audio, vision, missing='')

                    if return_sample_results:
                        ids.extend(batch_data['id'])
                        for item in features.keys():
                            features[item].append(outputs[item].cpu().detach().numpy())

                        all_labels_agreeableness.extend(labels_agreeableness.cpu().detach().tolist())
                        all_labels_openness.extend(labels_openness.cpu().detach().tolist())
                        all_labels_neuroticism.extend(labels_neuroticism.cpu().detach().tolist())
                        all_labels_extraversion.extend(labels_extraversion.cpu().detach().tolist())
                        all_labels_conscientiousness.extend(labels_conscientiousness.cpu().detach().tolist())

                        y_pred_agreeableness.append(outputs['agreeableness'].cpu())
                        y_pred_openness.append(outputs['openness'].cpu())
                        y_pred_neuroticism.append(outputs['neuroticism'].cpu())
                        y_pred_extraversion.append(outputs['extraversion'].cpu())
                        y_pred_conscientiousness.append(outputs['conscientiousness'].cpu())

                        preds_agreeableness = outputs["agreeableness"].cpu().detach().numpy()
                        preds_openness = outputs["openness"].cpu().detach().numpy()
                        preds_neuroticism = outputs["neuroticism"].cpu().detach().numpy()
                        preds_extraversion = outputs["extraversion"].cpu().detach().numpy()
                        preds_conscientiousness = outputs["conscientiousness"].cpu().detach().numpy()

                        sample_results_agreeableness.extend(preds_agreeableness.squeeze())
                        sample_results_openness.extend(preds_openness.squeeze())
                        sample_results_neuroticism.extend(preds_neuroticism.squeeze())
                        sample_results_extraversion.extend(preds_extraversion.squeeze())
                        sample_results_conscientiousness.extend(preds_conscientiousness.squeeze())

                    task_loss_agreeableness = self.criterion_agreeableness(outputs['agreeableness'].view(-1, 1),
                                                                           labels_agreeableness)
                    task_loss_openness = self.criterion_openness(outputs['openness'].view(-1, 1), labels_openness)
                    task_loss_neuroticism = self.criterion_neuroticism(outputs['neuroticism'].view(-1, 1), labels_neuroticism)
                    task_loss_extraversion = self.criterion_extraversion(outputs['extraversion'].view(-1, 1), labels_extraversion)
                    task_loss_conscientiousness = self.criterion_conscientiousness(outputs['conscientiousness'].view(-1, 1),
                                                                                   labels_conscientiousness)

                    eval_loss += task_loss_agreeableness.item()
                    eval_loss += task_loss_openness.item()
                    eval_loss += task_loss_neuroticism.item()
                    eval_loss += task_loss_extraversion.item()
                    eval_loss += task_loss_conscientiousness.item()

                    y_pred_agreeableness.append(outputs['agreeableness'].cpu())
                    y_pred_openness.append(outputs['openness'].cpu())
                    y_pred_neuroticism.append(outputs['neuroticism'].cpu())
                    y_pred_extraversion.append(outputs['extraversion'].cpu())
                    y_pred_conscientiousness.append(outputs['conscientiousness'].cpu())

                    y_true_agreeableness.append(labels_agreeableness.cpu())
                    y_true_openness.append(labels_openness.cpu())
                    y_true_neuroticism.append(labels_neuroticism.cpu())
                    y_true_extraversion.append(labels_extraversion.cpu())
                    y_true_conscientiousness.append(labels_conscientiousness.cpu())

        eval_loss = eval_loss / len(dataloader)

        pred_agreeableness, true_agreeableness = torch.cat(y_pred_agreeableness), torch.cat(y_true_agreeableness)
        pred_openness, true_openness = torch.cat(y_pred_openness), torch.cat(y_true_openness)
        pred_neuroticism, true_neuroticism = torch.cat(y_pred_neuroticism), torch.cat(y_true_neuroticism)
        pred_extraversion, true_extraversion = torch.cat(y_pred_extraversion), torch.cat(y_true_extraversion)
        pred_conscientiousness, true_conscientiousness = torch.cat(y_pred_conscientiousness), torch.cat(
            y_true_conscientiousness)

        eval_results_agreeableness = self.metrics(pred_agreeableness, true_agreeableness)
        eval_results_openness = self.metrics(pred_openness, true_openness)
        eval_results_neuroticism = self.metrics(pred_neuroticism, true_neuroticism)
        eval_results_extraversion = self.metrics(pred_extraversion, true_extraversion)
        eval_results_conscientiousness = self.metrics(pred_conscientiousness, true_conscientiousness)

        eval_results_agreeableness["Loss"] = round(eval_loss, 4)
        eval_results_openness["Loss"] = round(eval_loss, 4)
        eval_results_neuroticism["Loss"] = round(eval_loss, 4)
        eval_results_extraversion["Loss"] = round(eval_loss, 4)
        eval_results_conscientiousness["Loss"] = round(eval_loss, 4)

        logger.info(f"{mode}-({self.args.model_name}) Agreeableness >> {dict_to_str(eval_results_agreeableness)}")
        logger.info(f"{mode}-({self.args.model_name}) Openness>> {dict_to_str(eval_results_openness)}")
        logger.info(f"{mode}-({self.args.model_name}) Neuroticism>> {dict_to_str(eval_results_neuroticism)}")
        logger.info(f"{mode}-({self.args.model_name}) Extraversion>> {dict_to_str(eval_results_extraversion)}")
        logger.info(f"{mode}-({self.args.model_name}) Conscientiousness>> {dict_to_str(eval_results_conscientiousness)}")

        print(f"{mode}-({self.args.model_name}) "
              f">> loss: {round(eval_loss, 4)} "
              f">> Agreeableness: {dict_to_str(eval_results_agreeableness)}"
              f">> Openness: {dict_to_str(eval_results_openness)}"
              f">> Neuroticism: {dict_to_str(eval_results_neuroticism)}"
              f">> Extraversion: {dict_to_str(eval_results_extraversion)}"
              f">> Conscientiousness: {dict_to_str(eval_results_conscientiousness)}")

        if return_sample_results:
            eval_results_agreeableness["Ids"] = ids
            eval_results_agreeableness["SResults"] = sample_results_agreeableness
            eval_results_openness["Ids"] = ids
            eval_results_openness["SResults"] = sample_results_openness
            eval_results_neuroticism["Ids"] = ids
            eval_results_neuroticism["SResults"] = sample_results_neuroticism
            eval_results_extraversion["Ids"] = ids
            eval_results_extraversion["SResults"] = sample_results_extraversion
            eval_results_conscientiousness["Ids"] = ids
            eval_results_conscientiousness["SResults"] = sample_results_conscientiousness
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results_agreeableness['Features'] = features
            eval_results_agreeableness['Labels'] = all_labels_agreeableness
            eval_results_openness['Features'] = features
            eval_results_openness['Labels'] = all_labels_openness
            eval_results_neuroticism['Features'] = features
            eval_results_neuroticism['Labels'] = all_labels_neuroticism
            eval_results_extraversion['Features'] = features
            eval_results_extraversion['Labels'] = all_labels_extraversion
            eval_results_conscientiousness['Features'] = features
            eval_results_conscientiousness['Labels'] = all_labels_conscientiousness

        return eval_results_agreeableness, eval_results_openness, eval_results_neuroticism, eval_results_extraversion, eval_results_conscientiousness


    def reconstructed_modality(self, model, dataloader, mode="TEST"):
        model.eval()
        collected_data = []
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels_agreeableness = batch_data['labels']['agreeableness'].to(self.args.device)#.view(-1, 1)
                    labels_openness = batch_data['labels']['openness'].to(self.args.device)#.view(-1, 1)
                    labels_neuroticism = batch_data['labels']['neuroticism'].to(self.args.device)#.view(-1, 1)
                    labels_extraversion = batch_data['labels']['extraversion'].to(self.args.device)#.view(-1, 1)
                    labels_conscientiousness = batch_data['labels']['conscientiousness'].to(self.args.device)#.view(-1, 1)

                    outputs = model.generated_modality(text, audio, vision)

                    batch_size = vision.size(0)
                    for i in range(batch_size):
                        item = {
                            'original_text': outputs['original_text'][i].cpu(),
                            'original_vision': outputs['original_vision'][i].cpu(),
                            'original_audio': outputs['original_audio'][i].cpu(),
                            'ground_truth_text': outputs['ground_truth_text'][i].cpu(),
                            'ground_truth_vision': outputs['ground_truth_vision'][i].cpu(),
                            'ground_truth_audio': outputs['ground_truth_audio'][i].cpu(),
                            'generated_vision': outputs['generated_vision'][i].cpu(),
                            'generated_audio': outputs['generated_audio'][i].cpu(),
                            'agreeableness': labels_agreeableness[i].item(),
                            'openness': labels_openness[i].item(),
                            'neuroticism': labels_neuroticism[i].item(),
                            'extraversion': labels_extraversion[i].item(),
                            'conscientiousness': labels_conscientiousness[i].item(),
                        }
                        print(outputs['ground_truth_vision'][i].cpu() - outputs['generated_vision'][i].cpu())
                        collected_data.append(item)

        print(collected_data)
        # After all batches are processed, save to pickle
        save_path = 'reconstructed_outputs_'+str(mode)+'.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(collected_data, f)
        print(f"Saved reconstructed outputs to {save_path}")


    def plot_tsne_reconstructed_original(self):
        reconstructed_file = "reconstructed_outputs.pkl"

        original_feats_vision = []
        recon_feats_vision = []

        original_feats_audio = []
        recon_feats_audio = []

        with open(reconstructed_file, 'rb') as f:
            data = pickle.load(f)

        for data_dict in data:

            original_vision = data_dict['ground_truth_vision']
            recon_vision = data_dict['generated_vision']
            original_audio = data_dict['ground_truth_audio']
            recon_audio = data_dict['generated_audio']

            '''
            original_vision = data_dict['ground_truth_vision'].tolist()
            recon_vision = data_dict['generated_vision'].tolist()
            original_audio = data_dict['ground_truth_audio'].tolist()
            recon_audio = data_dict['generated_audio'].tolist()
            
            for i in range(len(recon_audio[0])):
                original_feats_vision.extend(original_vision[:, i])
                recon_feats_vision.extend(recon_vision[:, i])
                original_feats_audio.extend(original_audio[:, i])
                recon_feats_audio.extend(recon_audio[:, i])
            '''

            for i in range(recon_audio.shape[1]):
                original_feats_vision.append(original_vision[:, i].tolist())
                recon_feats_vision.append(recon_vision[:, i].tolist())
                original_feats_audio.append(original_audio[:, i].tolist())
                recon_feats_audio.append(recon_audio[:, i].tolist())

        print('len(original_feats_vision): ', len(original_feats_vision))
        # Combine and label
        X = np.vstack([original_feats_vision, recon_feats_vision])  # [N*2, 512]
        y = np.array([0] * len(original_feats_vision) + [1] * len(recon_feats_vision))  # 0 = original, 1 = recon

        # t-SNE embedding
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init='pca', random_state=42)
        X_2d = tsne.fit_transform(X)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], label="Original", alpha=0.6, s=20)
        plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], label="Reconstructed", alpha=0.6, s=20)
        plt.legend()
        plt.title("t-SNE of Original vs Reconstructed Visual Features")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('TSNE_Vision.png')
        plt.show()

        # Combine and label
        X = np.vstack([original_feats_audio, recon_feats_audio])  # [N*2, 512]
        y = np.array([0] * len(original_feats_audio) + [1] * len(recon_feats_audio))  # 0 = original, 1 = recon

        # t-SNE embedding
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init='pca', random_state=42)
        X_2d = tsne.fit_transform(X)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], label="Original", alpha=0.6, s=20)
        plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], label="Reconstructed", alpha=0.6, s=20)
        plt.legend()
        plt.title("t-SNE of Original vs Reconstructed Audio Features")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('TSNE_Audio.png')
        plt.show()


    def downstream_task(self, model, dataloader, reconstruction_modality='audio'):
        for param in model.score_l.parameters():
            param.requires_grad = False
        for param in model.score_v.parameters():
            param.requires_grad = False
        for param in model.score_a.parameters():
            param.requires_grad = False
        for param in model.cat_lv.parameters():
            param.requires_grad = False
        for param in model.cat_la.parameters():
            param.requires_grad = False
        for param in model.cat_va.parameters():
            param.requires_grad = False
        for param in model.rec_v.parameters():
            param.requires_grad = False
        for param in model.rec_l.parameters():
            param.requires_grad = False
        for param in model.rec_a.parameters():
            param.requires_grad = False
        for param in model.proj_l.parameters():
            param.requires_grad = False
        for param in model.proj_a.parameters():
            param.requires_grad = False
        for param in model.proj_v.parameters():
            param.requires_grad = False

        learning_rate = 0.0001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)
        # initilize results
        epochs, best_epoch = 0, 0
        best_valid = 1e8
        print('Started Training...')
        while True:
            epochs += 1
            # train
            y_pred_agreeableness = []
            y_pred_openness = []
            y_pred_neuroticism = []
            y_pred_extraversion = []
            y_pred_conscientiousness = []

            y_true_agreeableness = []
            y_true_openness = []
            y_true_neuroticism = []
            y_true_extraversion = []
            y_true_conscientiousness = []

            losses = []
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs

            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    if reconstruction_modality =='audio':
                        vision = batch_data['ground_truth_vision'].to(self.args.device)
                        audio = batch_data['generated_audio'].to(self.args.device)
                        text = batch_data['ground_truth_text'].to(self.args.device)
                    elif reconstruction_modality == 'vision':
                        vision = batch_data['generated_vision'].to(self.args.device)
                        audio = batch_data['ground_truth_audio'].to(self.args.device)
                        text = batch_data['ground_truth_text'].to(self.args.device)
                    labels_agreeableness = batch_data['labels']['agreeableness'].to(self.args.device)  # .view(-1, 1)
                    labels_openness = batch_data['labels']['openness'].to(self.args.device)  # .view(-1, 1)
                    labels_neuroticism = batch_data['labels']['neuroticism'].to(self.args.device)  # .view(-1, 1)
                    labels_extraversion = batch_data['labels']['extraversion'].to(self.args.device)  # .view(-1, 1)
                    labels_conscientiousness = batch_data['labels']['conscientiousness'].to(self.args.device)  # .view(-1, 1)

                    outputs = model.downstream_task_trainer(text, audio, vision)

                    # compute loss

                    task_loss_agreeableness = self.criterion_agreeableness(outputs['agreeableness'].view(-1, 1),
                                                                           labels_agreeableness)
                    task_loss_openness = self.criterion_openness(outputs['openness'].view(-1, 1), labels_openness)
                    task_loss_neuroticism = self.criterion_neuroticism(outputs['neuroticism'].view(-1, 1),
                                                                       labels_neuroticism)
                    task_loss_extraversion = self.criterion_extraversion(outputs['extraversion'].view(-1, 1),
                                                                         labels_extraversion)
                    task_loss_conscientiousness = self.criterion_conscientiousness(outputs['conscientiousness'].view(-1, 1), labels_conscientiousness)

                    combine_loss = task_loss_agreeableness + task_loss_openness + task_loss_neuroticism + task_loss_extraversion + task_loss_conscientiousness

                    # backward
                    combine_loss.backward()
                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad],
                                                  self.args.grad_clip)
                    # store results
                    train_loss += combine_loss.item()

                    y_pred_agreeableness.append(outputs['agreeableness'].cpu())
                    y_pred_openness.append(outputs['openness'].cpu())
                    y_pred_neuroticism.append(outputs['neuroticism'].cpu())
                    y_pred_extraversion.append(outputs['extraversion'].cpu())
                    y_pred_conscientiousness.append(outputs['conscientiousness'].cpu())

                    y_true_agreeableness.append(labels_agreeableness.cpu())
                    y_true_openness.append(labels_openness.cpu())
                    y_true_neuroticism.append(labels_neuroticism.cpu())
                    y_true_extraversion.append(labels_extraversion.cpu())
                    y_true_conscientiousness.append(labels_conscientiousness.cpu())

                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()
            train_loss = train_loss / len(dataloader['train'])

            pred_agreeableness, true_agreeableness = torch.cat(y_pred_agreeableness), torch.cat(y_true_agreeableness)
            pred_openness, true_openness = torch.cat(y_pred_openness), torch.cat(y_true_openness)
            pred_neuroticism, true_neuroticism = torch.cat(y_pred_neuroticism), torch.cat(y_true_neuroticism)
            pred_extraversion, true_extraversion = torch.cat(y_pred_extraversion), torch.cat(y_true_extraversion)
            pred_conscientiousness, true_conscientiousness = torch.cat(y_pred_conscientiousness), torch.cat(
                y_true_conscientiousness)

            train_results_agreeableness = self.metrics(pred_agreeableness, true_agreeableness)
            train_results_openness = self.metrics(pred_openness, true_openness)
            train_results_neuroticism = self.metrics(pred_neuroticism, true_neuroticism)
            train_results_extraversion = self.metrics(pred_extraversion, true_extraversion)
            train_results_conscientiousness = self.metrics(pred_conscientiousness, true_conscientiousness)

            logger.info(
                f"TRAIN DOWNSTREAM-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> loss: {round(train_loss, 4)} "
                f">> Agreeableness: {dict_to_str(train_results_agreeableness)}"
                f">> Openness: {dict_to_str(train_results_openness)}"
                f">> Neuroticism: {dict_to_str(train_results_neuroticism)}"
                f">> Extraversion: {dict_to_str(train_results_extraversion)}"
                f">> Conscientiousness: {dict_to_str(train_results_conscientiousness)}"
            )
            print(f"TRAIN DOWNSTREAM-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                  f">> loss: {round(train_loss, 4)} \n"
                  f">> Agreeableness: {dict_to_str(train_results_agreeableness)}\n"
                  f">> Openness: {dict_to_str(train_results_openness)}\n"
                  f">> Neuroticism: {dict_to_str(train_results_neuroticism)}\n"
                  f">> Extraversion: {dict_to_str(train_results_extraversion)}\n"
                  f">> Conscientiousness: {dict_to_str(train_results_conscientiousness)}\n")

            # validation
            val_loss = self.downstream_task_validation(model, dataloader['valid'], mode="VAL")
            test_loss = self.downstream_task_validation(model, dataloader['test'], mode="TEST")

            scheduler.step(val_loss)
            # save each epoch model
            # model_save_path = 'pt/' + str(epochs) + '.pth'  #Commented by Rishabh to save on space
            # torch.save(model.state_dict(), model_save_path)  #Commented by Rishabh to save on space


            model.to(self.args.device)

            # save best model
            isBetter = val_loss <= (best_valid - 1e-6)
            if isBetter:
                best_valid, best_epoch = val_loss, epochs
                # save model
                print('Saving the model')
                torch.save(model.cpu().state_dict(), 'pt/remodiff_downstream_' + str(reconstruction_modality) + '.pth')
                print('Model saved')
                model.to(self.args.device)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                print('Completed Training...')
                print('Best Epoch: ', best_epoch, ' Best Validation Loss: ', best_valid)
                return  # epoch_results if return_epoch_results else None

    def downstream_task_validation(self, model, dataloader, mode="VAL", reconstruction_modality='audio'):
        model.eval()

        y_pred_agreeableness = []
        y_pred_openness = []
        y_pred_neuroticism = []
        y_pred_extraversion = []
        y_pred_conscientiousness = []

        y_true_agreeableness = []
        y_true_openness = []
        y_true_neuroticism = []
        y_true_extraversion = []
        y_true_conscientiousness = []

        eval_loss = 0.0

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    if reconstruction_modality == 'audio':
                        vision = batch_data['ground_truth_vision'].to(self.args.device)
                        audio = batch_data['generated_audio'].to(self.args.device)
                        text = batch_data['ground_truth_text'].to(self.args.device)
                    elif reconstruction_modality == 'vision':
                        vision = batch_data['generated_vision'].to(self.args.device)
                        audio = batch_data['ground_truth_audio'].to(self.args.device)
                        text = batch_data['ground_truth_text'].to(self.args.device)
                    labels_agreeableness = batch_data['labels']['agreeableness'].to(
                        self.args.device)  # .view(-1, 1)
                    labels_openness = batch_data['labels']['openness'].to(self.args.device)  # .view(-1, 1)
                    labels_neuroticism = batch_data['labels']['neuroticism'].to(self.args.device)  # .view(-1, 1)
                    labels_extraversion = batch_data['labels']['extraversion'].to(self.args.device)  # .view(-1, 1)
                    labels_conscientiousness = batch_data['labels']['conscientiousness'].to(
                        self.args.device)  # .view(-1, 1)

                    outputs = model.downstream_task_trainer(text, audio, vision)

                    task_loss_agreeableness = self.criterion_agreeableness(outputs['agreeableness'].view(-1, 1),
                                                                           labels_agreeableness)
                    task_loss_openness = self.criterion_openness(outputs['openness'].view(-1, 1), labels_openness)
                    task_loss_neuroticism = self.criterion_neuroticism(outputs['neuroticism'].view(-1, 1),
                                                                       labels_neuroticism)
                    task_loss_extraversion = self.criterion_extraversion(outputs['extraversion'].view(-1, 1),
                                                                         labels_extraversion)
                    task_loss_conscientiousness = self.criterion_conscientiousness(
                        outputs['conscientiousness'].view(-1, 1),
                        labels_conscientiousness)

                    eval_loss += task_loss_agreeableness.item()
                    eval_loss += task_loss_openness.item()
                    eval_loss += task_loss_neuroticism.item()
                    eval_loss += task_loss_extraversion.item()
                    eval_loss += task_loss_conscientiousness.item()

                    y_pred_agreeableness.append(outputs['agreeableness'].cpu())
                    y_pred_openness.append(outputs['openness'].cpu())
                    y_pred_neuroticism.append(outputs['neuroticism'].cpu())
                    y_pred_extraversion.append(outputs['extraversion'].cpu())
                    y_pred_conscientiousness.append(outputs['conscientiousness'].cpu())

                    y_true_agreeableness.append(labels_agreeableness.cpu())
                    y_true_openness.append(labels_openness.cpu())
                    y_true_neuroticism.append(labels_neuroticism.cpu())
                    y_true_extraversion.append(labels_extraversion.cpu())
                    y_true_conscientiousness.append(labels_conscientiousness.cpu())

        eval_loss = eval_loss / len(dataloader)

        pred_agreeableness, true_agreeableness = torch.cat(y_pred_agreeableness), torch.cat(y_true_agreeableness)
        pred_openness, true_openness = torch.cat(y_pred_openness), torch.cat(y_true_openness)
        pred_neuroticism, true_neuroticism = torch.cat(y_pred_neuroticism), torch.cat(y_true_neuroticism)
        pred_extraversion, true_extraversion = torch.cat(y_pred_extraversion), torch.cat(y_true_extraversion)
        pred_conscientiousness, true_conscientiousness = torch.cat(y_pred_conscientiousness), torch.cat(
            y_true_conscientiousness)

        eval_results_agreeableness = self.metrics(pred_agreeableness, true_agreeableness)
        eval_results_openness = self.metrics(pred_openness, true_openness)
        eval_results_neuroticism = self.metrics(pred_neuroticism, true_neuroticism)
        eval_results_extraversion = self.metrics(pred_extraversion, true_extraversion)
        eval_results_conscientiousness = self.metrics(pred_conscientiousness, true_conscientiousness)

        eval_results_agreeableness["Loss"] = round(eval_loss, 4)
        eval_results_openness["Loss"] = round(eval_loss, 4)
        eval_results_neuroticism["Loss"] = round(eval_loss, 4)
        eval_results_extraversion["Loss"] = round(eval_loss, 4)
        eval_results_conscientiousness["Loss"] = round(eval_loss, 4)

        logger.info(f"{mode}-({self.args.model_name}) Agreeableness >> {dict_to_str(eval_results_agreeableness)}")
        logger.info(f"{mode}-({self.args.model_name}) Openness>> {dict_to_str(eval_results_openness)}")
        logger.info(f"{mode}-({self.args.model_name}) Neuroticism>> {dict_to_str(eval_results_neuroticism)}")
        logger.info(f"{mode}-({self.args.model_name}) Extraversion>> {dict_to_str(eval_results_extraversion)}")
        logger.info(
            f"{mode}-({self.args.model_name}) Conscientiousness>> {dict_to_str(eval_results_conscientiousness)}")

        print(f"{mode}-({self.args.model_name}) "
              f">> loss: {round(eval_loss, 4)} \n"
              f">> Agreeableness: {dict_to_str(eval_results_agreeableness)}\n"
              f">> Openness: {dict_to_str(eval_results_openness)}\n"
              f">> Neuroticism: {dict_to_str(eval_results_neuroticism)}\n"
              f">> Extraversion: {dict_to_str(eval_results_extraversion)}\n"
              f">> Conscientiousness: {dict_to_str(eval_results_conscientiousness)}\n")

        return eval_loss