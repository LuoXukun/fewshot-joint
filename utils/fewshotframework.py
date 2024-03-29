#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.10.21

import os
import torch
import torch.nn as nn

#from torch.nn import DataParallel
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

class FewshotJointFramework:
    def __init__(self, args):
        """ FewShot Framework for Joint Extraction of Entities and Relations. """
        self.train_data_loader = args.train_data_loader
        self.valid_data_loader = args.valid_data_loader
        #self.test_data_loader = args.test_data_loader
        self.logger = args.logger

        self.pre_train_data_loader = args.pre_train_data_loader
        self.pre_valid_data_loader = args.pre_valid_data_loader

    def __load_model__(self, ckpt):
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            self.logger.info("Successfully loaded checkpoint '{}'".format(ckpt))
            return checkpoint
        else:
            self.logger.critical("No checkpoint found at '{}'".format(ckpt))
            exit()
    
    def train(self, args):
        """ 
            Training step.

            Args:

                model:                  Few shot model.
                model_type:             Few shot model type.
                metrics_calculator:     Few shot metrics calculator.
                batch_size:             Batch size.
                N:                      Number of classes for each batch.
                K:                      Number of instances of each class in support set.
                Q:                      Number of instances of each class in query set.
                learning_rate:          Initial learning rate.
                train_iter:             Num of iterations of training.
                val_iter:               Num of iterations of validating.
                val_step:               Validate every val_step steps.
                report_step:            Validate every train_step steps.
                load_ckpt:              Directory of checkpoints for loading. Default: None.
                save_ckpt:              Directory of checkpoints for saving. Default: None.
                warmup_step:            Warming up steps.
                grad_iter:              Iter of gradient descent. Default: 1.
                tag_seqs_num:           Tag seqs number.
                use_fp16:               If use fp16.
            
            Returns:
        """
        self.logger.info("Start training...")

        # Load model.
        if args.load_ckpt:
            self.logger.info("Loading checkpoint '{}'...".format(args.load_ckpt))
            state_dict = self.__load_model__(args.load_ckpt)['state_dict']
            own_state = args.model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    self.logger.warning("Ignore {}".format(name))
                    continue
                self.logger.info("Loading {} from {}".format(name, args.load_ckpt))
                own_state[name].copy_(param)
        start_iter = 0

        # For simplicity, we use DataParallel wrapper to use multiple GPUs.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """ if torch.cuda.device_count() > 1:
            self.logger.info("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
            args.model = nn.DataParallel(args.model)
            #args.model = args.model.module """
        args.model = args.model.to(device)

        # Init optimizer.
        self.logger.info("Use Bert Optim...")
        parameters_to_optimize = list(args.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(parameters_to_optimize, lr=args.learning_rate, correct_bias=False)

        if args.use_fp16:
            from apex import amp
            args.model, optimizer = amp.initialize(args.model, optimizer, opt_level="O1")
        
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_step/args.grad_iter), num_training_steps=int(args.train_iter/args.grad_iter))

        # Training.
        best_f1, iter_loss, iter_accs = 0.0, 0.0, None
        args.model.train()

        for it in range(start_iter, start_iter + args.train_iter):
            support, query = next(self.train_data_loader)
            support["tags"] = [torch.stack(support["tags"][i], 0).to(device) for i in range(args.tag_seqs_num)]

            if args.model_type in ["few-tplinker", "few-tplinker-plus"]:
                while torch.max(support["tags"][0]) != 1 or torch.max(support["tags"][1]) != 1:
                    #args.logger.warning("Invalid samples, get new samples...")
                    support, query = next(self.train_data_loader)
                    support["tags"] = [torch.stack(support["tags"][i], 0).to(device) for i in range(args.tag_seqs_num)]

            for k in support:
                if k != "tags" and k != "samples_num" and k != "rel_id" and k != "sample":
                    support[k] = support[k].to(device)
                    query[k] = query[k].to(device)
            label = [torch.stack(query["tags"][i], 0).to(device) for i in range(args.tag_seqs_num)]
            
            #print("support: {}, query: {}".format(support["src_ids"].shape, query["src_ids"].shape))
            logits, preds = args.model(support, query)
            #print("support: {}, query: {}, logits: {}, label: {}, samples_num: {}".format(support["src_ids"].shape, query["src_ids"].shape, logits[0].shape, label[0].shape, query["samples_num"]))
            assert logits[0].shape[:2] == label[0].shape[:2]
            #assert logits[1].shape[:2] == label[1].shape[:2]
            #assert logits[2].shape[:2] == label[2].shape[:2]
            loss = args.model.loss(logits, label) / float(args.grad_iter)
            accs = args.metrics_calculator.get_accs(preds, label)

            if args.use_fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if it % args.grad_iter == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            iter_loss += loss.data.item()

            if not iter_accs:
                iter_accs = {key: 0.0 for key, value in accs.items()}
            for key, value in accs.items():
                iter_accs[key] += value
            
            if (it + 1) % args.report_step == 0:
                batch_format = "TRAIN -- Step: {}, loss: {},".format(it + 1, iter_loss / float(args.report_step))
                for key, value in iter_accs.items():
                    batch_format += " {}: {},".format(key, value / float(args.report_step))
                batch_format = batch_format[:-1]
                self.logger.info(batch_format)
                iter_loss = 0.0; iter_accs = None

            # Validation.
            if(it + 1) % args.val_step == 0:
                val_f1 = self.eval(args.model, args.metrics_calculator, args.val_iter, device, args.tag_seqs_num, args.model_type)
                args.model.train()
                if val_f1 > best_f1 or (it + 1) == args.val_step:
                    if args.save_ckpt:
                        self.logger.info("Better checkpoint! Saving...")
                        torch.save({"state_dict": args.model.state_dict()}, args.save_ckpt)
                    best_f1 = val_f1
        
        # Testing.
        self.logger.info("Start Testing...")
        if args.save_ckpt:
            test_f1 = self.eval(args.model, args.metrics_calculator, args.val_iter, device, args.tag_seqs_num, args.model_type, ckpt=args.save_ckpt)
        else:
            self.logger.warning("There is no a saved checkpoint path, so we cannnot test on the best model!")
        
        self.logger.info("Finish Training")
    
    def eval(self, model, metrics_calculator, eval_iter, device, tag_seqs_num, model_type, ckpt=None):
        '''
            Validation.
            Args:
                model:                  FewShotREModel instance.
                metrics_calculator:     Few shot metrics calculator.
                eval_iter:              Num of iterations.
                device:                 CPU or GPU.
                tag_seqs_num:           Tag seqs number.
                model_type:             Few shot model type.
                ckpt:                   Checkpoint path. Set as None if using current model parameters.
            Returns: 
                f1
        '''
        self.logger.info("Evaluation...")
        model.eval()

        eval_data_loader = self.valid_data_loader
        total_iter = eval_iter

        if ckpt is not None:
            self.logger.info("Loading best checkpoint '{}'...".format(ckpt))
            state_dict = self.__load_model__(ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            total_iter = eval_iter * 4
        
        pred_cnt, gold_cnt, correct_cnt = 0, 0, 0
        
        with torch.no_grad():
            for it in range(total_iter):
                support, query = next(eval_data_loader)
                support["tags"] = [torch.stack(support["tags"][i], 0).to(device) for i in range(tag_seqs_num)]

                if model_type in ["few-tplinker", "few-tplinker-plus"]:
                    while torch.max(support["tags"][0]) != 1 or torch.max(support["tags"][1]) != 1:
                        #self.logger.warning("Invalid samples in iter {}, get new samples...".format(it + 1))
                        support, query = next(eval_data_loader)
                        support["tags"] = [torch.stack(support["tags"][i], 0).to(device) for i in range(tag_seqs_num)]

                for k in support:
                    if k != "tags" and k != "samples_num" and k != "rel_id" and k != "sample":
                        support[k] = support[k].to(device)
                        query[k] = query[k].to(device)
                #print("tags: ", support["tags"][0].size())
                samples = query["sample"]
                
                try:
                    _, preds = model.inference(support, query)
                except:
                    #self.logger.Warning("There is no inference function in the model, using forward function...")
                    _, preds = model(support, query)
                tmp_pred_cnt, tmp_gold_cnt, tmp_correct_cnt = metrics_calculator.get_rel_pgc(samples, preds)
                pred_cnt += tmp_pred_cnt; gold_cnt += tmp_gold_cnt; correct_cnt += tmp_correct_cnt
            
            prec, rec, f1 = self.get_prf(pred_cnt, gold_cnt, correct_cnt)

            self.logger.critical(
                "EVAL -- Eval_iter: {}, Pred: {}, Gold: {}, Correct: {}, Prec: {}, Rec: {}, F1: {}".format(
                    total_iter, pred_cnt, gold_cnt, correct_cnt, prec, rec, f1
                )
            )
        
        return f1
    
    def pretrain(self, args):
        """ 
            Pre-training step.

            Args:

                pre_model:              Pre-train model.
                model_type:             Few shot model type.
                metrics_calculator:     Few shot metrics calculator.
                pre_batch_size:         Batch size.
                pre_learning_rate:      Initial pre-train learning rate.
                pre_train_epoch:        Num of epochs of training.
                pre_val_epoch:          Validate every pre-train val_step epochs.
                pre_report_step:        Validate every pre-train train_step steps.
                pre_ckpt:               Directory of pre-train checkpoints for saving. Default: None.
                pre_warmup_rate:        Warming up rate of pre-training.
                pre_grad_iter:          Iter of gradient descent. Default: 1.
                tag_seqs_num:           Tag seqs number.
                use_fp16:               If use fp16.
            
            Returns:
        """
        self.logger.info("Start pre-training...")

        # Load model.
        if args.pre_ckpt:
            if os.path.exists(args.pre_ckpt):
                self.logger.info("Loading checkpoint '{}'...".format(args.pre_ckpt))
                state_dict = self.__load_model__(args.pre_ckpt)['state_dict']
                own_state = args.pre_model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        self.logger.warning("Ignore {}".format(name))
                        continue
                    self.logger.info("Loading {} from {}".format(name, args.pre_ckpt))
                    own_state[name].copy_(param)
                # Init pre-train model. For simplicity, we do not use DP or DDP.
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                args.pre_model = args.pre_model.to(device)
                self.logger.info("Evaluating the checkpoint of pre-trained model.")
                val_f1 = self.pre_eval(args.pre_model, args.metrics_calculator, device, args.tag_seqs_num)
                #exit()

        # Init pre-train model. For simplicity, we do not use DP or DDP.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.pre_model = args.pre_model.to(device)

        # Init optimizer.
        self.logger.info("Use Bert Optim...")
        parameters_to_optimize = list(args.pre_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(parameters_to_optimize, lr=args.pre_learning_rate, correct_bias=False)

        if args.use_fp16:
            from apex import amp
            args.pre_model, optimizer = amp.initialize(args.pre_model, optimizer, opt_level="O1")
        
        total_steps = int(args.pre_train_epoch * len(self.pre_train_data_loader.dataset) / args.pre_batch_size) + 1
        warmup_steps = int(total_steps * args.pre_warmup_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        # Pre-training.
        best_f1 = 0.0

        for epoch_id in range(1, args.pre_train_epoch + 1):
            iter_loss, iter_accs = 0.0, None
            args.pre_model.train()

            # Train.
            for batch_id, batch in enumerate(self.pre_train_data_loader):
                for k in batch:
                    if k != "tags" and k != "rel_id" and k != "sample":
                        batch[k] = batch[k].to(device)
                batch["tags"] = [torch.stack(batch["tags"][i], 0).to(device) for i in range(args.tag_seqs_num)]
                
                logits, preds = args.pre_model(batch)
                #print("logits: {}, tags: {}".format(logits[0].shape, batch["tags"][0].shape))
                assert logits[0].shape[:2] == batch["tags"][0].shape[:2]
                if (batch_id + 1) % args.pre_report_step == 0 or batch_id == len(self.pre_train_data_loader.dataset) - 1:
                    loss = args.pre_model.loss(logits, batch["tags"], quiet=False) / float(args.pre_grad_iter)
                else:
                    loss = args.pre_model.loss(logits, batch["tags"]) / float(args.pre_grad_iter)
                accs = args.metrics_calculator.get_accs(preds, batch["tags"])
                iter_loss += loss.data.item()

                if args.use_fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                if batch_id % args.pre_grad_iter == 0 or batch_id == len(self.pre_train_data_loader.dataset) - 1:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if not iter_accs:
                    iter_accs = {key: 0.0 for key, value in accs.items()}
                for key, value in accs.items():
                    iter_accs[key] += value
                
                if (batch_id + 1) % args.pre_report_step == 0 or batch_id == len(self.pre_train_data_loader.dataset) - 1:
                    batch_format = "PRE_TRAIN -- Epoch: {}, Step: {}, loss: {},".format(epoch_id, batch_id + 1, iter_loss / float(args.pre_report_step))
                    for key, value in iter_accs.items():
                        batch_format += " {}: {},".format(key, value / float(args.pre_report_step))
                    batch_format = batch_format[:-1]
                    self.logger.info(batch_format)
                    iter_loss = 0.0; iter_accs = None
            
            # Valid.
            if epoch_id % args.pre_val_epoch == 0 or epoch_id == args.pre_train_epoch:
                val_f1 = self.pre_eval(args.pre_model, args.metrics_calculator, device, args.tag_seqs_num)
                if val_f1 > best_f1 or epoch_id == args.pre_val_epoch:
                    if args.pre_ckpt:
                        self.logger.info("Better checkpoint! Saving...")
                        torch.save({"state_dict": args.pre_model.state_dict()}, args.pre_ckpt)
                    best_f1 = val_f1
        
        self.logger.info("Finish Pre-training...")
    
    def pre_eval(self, pre_model, metrics_calculator, device, tag_seqs_num):
        '''
            Validation for pre-train.
            Args:
                pre_model:              Pre-train REModel instance.
                metrics_calculator:     Few shot metrics calculator.
                device:                 CPU or GPU.
                tag_seqs_num:           Tag seqs number.
                model_type:             Few shot model type.
            Returns: 
                f1
        '''
        self.logger.info("Pretrain evaluation...")
        pre_model.eval()

        pred_cnt, gold_cnt, correct_cnt = 0, 0, 0

        with torch.no_grad():
            for batch_id, batch in enumerate(self.pre_valid_data_loader):
            #for batch_id, batch in tqdm(enumerate(self.pre_train_data_loader)):
                #print("batch_id: ", batch_id)
                for k in batch:
                    if k != "tags" and k != "rel_id" and k != "sample":
                        batch[k] = batch[k].to(device)
                batch["tags"] = [torch.stack(batch["tags"][i], 0).to(device) for i in range(tag_seqs_num)]

                logits, preds = pre_model(batch)

                tmp_pred_cnt, tmp_gold_cnt, tmp_correct_cnt = metrics_calculator.get_rel_pgc(batch["sample"], preds)
                pred_cnt += tmp_pred_cnt; gold_cnt += tmp_gold_cnt; correct_cnt += tmp_correct_cnt
            
            prec, rec, f1 = self.get_prf(pred_cnt, gold_cnt, correct_cnt)
        
        self.logger.critical(
            "PRE_TRAIN EVAL -- Eval_instances: {}, Pred: {}, Gold: {}, Correct: {}, Prec: {}, Rec: {}, F1: {}".format(
                len(self.pre_valid_data_loader), pred_cnt, gold_cnt, correct_cnt, prec, rec, f1
            )
        )

        return f1

    def test(self, args):
        """ 
            Simple test.
        """
        eval_data_loader = self.valid_data_loader

        self.logger.info("Start testing...")

        # Load model.
        if args.load_ckpt:
            self.logger.info("Loading checkpoint '{}'...".format(args.load_ckpt))
            state_dict = self.__load_model__(args.load_ckpt)['state_dict']
            own_state = args.model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    self.logger.warning("Ignore {}".format(name))
                    continue
                own_state[name].copy_(param)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.model = args.model.to(device)

        # Testing.
        pred_cnt, gold_cnt, correct_cnt = 0, 0, 0
        args.model.eval()

        with torch.no_grad():
            for it in range(args.val_iter):
                support, query = next(eval_data_loader)
                support["tags"] = [torch.stack(support["tags"][i], 0).to(device) for i in range(args.tag_seqs_num)]

                if args.model_type in ["few-tplinker", "few-tplinker-plus"]:
                    while torch.max(support["tags"][0]) != 1 or torch.max(support["tags"][1]) != 1:
                        args.logger.warning("Invalid samples in iter {}, get new samples...".format(it + 1))
                        support, query = next(eval_data_loader)
                        support["tags"] = [torch.stack(support["tags"][i], 0).to(device) for i in range(args.tag_seqs_num)]

                for k in support:
                    if k != "tags" and k != "samples_num" and k != "rel_id" and k != "sample":
                        support[k] = support[k].to(device)
                        query[k] = query[k].to(device)
                #print("tags: ", support["tags"][0].size())
                samples = query["sample"]
                
                try:
                    _, preds = args.model.inference(support, query)
                except:
                    self.logger.Warning("There is no inference function in the model, using forward function...")
                    _, preds = args.model(support, query)

                tmp_pred_cnt, tmp_gold_cnt, tmp_correct_cnt = args.metrics_calculator.get_rel_pgc(samples, preds)
                pred_cnt += tmp_pred_cnt; gold_cnt += tmp_gold_cnt; correct_cnt += tmp_correct_cnt
            
            prec, rec, f1 = self.get_prf(pred_cnt, gold_cnt, correct_cnt)

            self.logger.critical(
                "EVAL -- Eval_iter: {}, Pred: {}, Gold: {}, Correct: {}, Prec: {}, Rec: {}, F1: {}".format(
                    args.val_iter, pred_cnt, gold_cnt, correct_cnt, prec, rec, f1
                )
            )
        
        self.logger.info("Finish Testing.")

    def get_prf(self, pred, gold, correct):
        mini_num = 1e-10
        precision = correct / (pred + mini_num)
        recall = correct / (gold + mini_num)
        f1 = 2 * precision * recall / (precision + recall + mini_num)
        return precision, recall, f1
