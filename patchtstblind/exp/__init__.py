import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from patchtstblind.utils.train import EarlyStopping
from patchtstblind.utils.data_factory import data_provider
from patchtstblind.utils.models import get_criterion, \
                                          get_model, \
                                          get_optim, \
                                          get_scheduler, \
                                          compute_loss, \
                                          model_update

# Logger
import neptune

# Timing
import time

class Experiment:
    def __init__(self, args):
        self.args = args

    def run(self):
        torch.manual_seed(self.args.seed)
        self.init_logger()

        self.init_devices()
        self.init_model()
        self.count_parameters()
        self.init_optimizer()
        self.init_dataloaders()

        self.supervised_train()

        if self.logger_run is not None: 
            self.logger_run.stop()


    def init_devices(self):
        """
        Initialize CUDA (or MPS) devices.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device(s) initialized to: {self.device}")

    def init_dataloaders(self):
        """
        Initialize the dataloaders.
        """

        # SL dataloaders
        self.train_data, self.train_loader = data_provider(self.args, flag="train")
        self.val_data, self.val_loader = data_provider(self.args, flag="val")
        self.test_data, self.test_loader = data_provider(self.args, flag="test")

            # SL Logging
        if self.logger_run is not None:
            self.logger_run["train/num_examples"] = len(self.train_data)
            self.logger_run["validation/num_examples"] = len(self.val_data)
            self.logger_run["test/num_examples"] = len(self.test_data)
        
        print(f"{len(self.train_data)} training examples, {len(self.val_data)} validation examples, {len(self.test_data)} test examples.")

        print("Dataloaders initialized.")

    def init_model(self):
        """
            Initialize the model.
        """
        self.model = get_model(self.args)
        self.model.to(self.device)

    def init_optimizer(self):
        """
            Initialize the optimizer
        """
        self.optimizer = get_optim(self.args, self.model, self.args.optim_type)

    def init_logger(self):
        """
            Initialize the logger
        """

        # Initialize Neptune run with the time-based ID
        try:
            self.logger_run = neptune.init_run(project=self.args.project_name,
                                           api_token=self.args.api_token,
                                           custom_run_id=self.args.run_id)
            
        except neptune.common.exceptions.NeptuneInvalidApiTokenException:
            self.logger_run = None
            print("NO API KEY PROVIDED...")
            
        # Log parameters
        if self.logger_run is not None:
            self.logger_run["parameters"] = self.args
            print("Logger initialized.")


    def init_earlystopping(self, path : str):

        self.early_stopping = EarlyStopping(patience=self.args.patience,
                                           verbose=self.args.verbose,
                                           delta=self.args.patience,
                                           path=path)

    def train(self, model, model_id, optimizer, train_loader, best_model_path, criterion, val_loader=None, scheduler=None, flag="sl", ema=None, mae=False):
        """
            Trains a model.

        Args:
            model (nn.Module): The model to train.
            model_id (str): The model ID.
            optimizer (torch.optim): The optimizer to use.
            train_loader (torch.utils.data.DataLoader): The training data.
            best_model_path (str): The path to save the best model.
            criterion (torch.nn): The loss function.
            val_loader (torch.utils.data.DataLoader): The validation data.
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
            flag (str): The type of learning. Options: "sl".
        """
        print("Training...")
        if self.args.early_stopping:
            self.init_earlystopping(best_model_path)

        model.train()

        # <--------------- Training --------------->
        for epoch in range(eval(f"self.args.{flag}_epochs")):
            running_loss = 0
            start_time = time.time()
            alpha = 0
            for i, (x, y, _, _) in enumerate(train_loader):
                optimizer.zero_grad()
                x = x.permute(0, 2, 1).float().to(self.device) # (batch, seq_len, num_channels) -> (batch, num_channels, seq_len)
                y = y[:, -self.args.pred_len:].permute(0, 2, 1).float().to(self.device) # Get last pred_len values in y  and reshape to (batch, num_channels, seq_len)
                output = model(x)
                loss = compute_loss(output, y, criterion, model_id)

                # Running loss
                running_loss+=loss.item()

                # Update model parameters
                alpha = next(ema) if ema else 0.966
                model_update(model, loss, optimizer, model_id, alpha)

                # Update learning rate (optional)
                if scheduler:
                    scheduler.step()

                # Timing
                if (i+1) % 100 == 0:
                    end_time = time.time()
                    print(f"Batch {i+1}/{len(train_loader)}: {end_time - start_time }s. Loss: {loss.item()}")
                    start_time = time.time()

            # Average Loss + Logging
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch}. Training loss: {epoch_loss:.6f}.")
            if self.logger_run is not None: self.logger_run[f"{flag}_train/loss"].append(epoch_loss)


            # <--------------- Validation --------------->
            if val_loader:
                stats = iter(self.evaluate(model=model, model_id=model_id, loader=val_loader, criterion=criterion, flag=flag, mae=mae))
                val_loss = next(stats)
                print(f"Epoch: {epoch}, Validation Loss: {val_loss:.6f}")
                if self.logger_run is not None: self.logger_run[f"{flag}_validation/loss"].append(val_loss)

                if mae:
                    val_mae = next(stats)
                    print(f"Epoch: {epoch}, Validation MAE: {val_mae:.6f}")
                    if self.logger_run is not None: self.logger_run[f"{flag}_validation/mae"].append(val_mae)

                # Print & log validation accuracy (optional)
                if self.args.acc:
                    val_acc = next(stats)
                    print(f"Epoch: {epoch}, Validation Accuracy: {val_acc:.3f}")
                    if self.logger_run is not None: self.logger_run[f"{flag}_validation/accuracy"].append(val_acc)

                

                # Save best model + Early Stopping
                if self.args.early_stopping:
                    self.early_stopping(val_loss, model)
                else:
                    path_dir = os.path.abspath(os.path.dirname(best_model_path))
                    if not os.path.isdir(path_dir):
                        os.makedirs(path_dir)
                    print(f"Saving Model Weights in {best_model_path}...")
                    torch.save(model.state_dict(), best_model_path)

            if self.args.early_stopping:
                if self.early_stopping.early_stop:
                    print("EarlyStopping activated, ending training.")
                    break

            if self.logger_run is not None:
                self.logger_run[f"model_checkpoints/{model_id}_{flag}"].upload(best_model_path)


    def supervised_train(self):
        """
            Train the model in supervised mode.
        """

        # Get supervised criterion
        self.criterion = get_criterion(self.args.criterion)

        # Get supervised scheduler
        if self.args.scheduler != "None":
            self.scheduler = get_scheduler(self.args, self.args.scheduler, "supervised", self.optimizer)
        else:
            self.scheduler = None
        print("Start Supervised Training.")

        # Supervised training
        self.train(model=self.model,
                   model_id=self.args.model_id,
                   optimizer=self.optimizer,
                   train_loader=self.train_loader,
                   best_model_path=os.path.join(self.args.best_model_path, f"{self.args.model_id}_supervised.pth"),
                   criterion=self.criterion,
                   val_loader=self.val_loader,
                   scheduler=self.scheduler,
                   flag="sl",
                   mae=True)
        print("Start Supervised Testing.")

        # Test model
        self.test(model=self.model,
                  model_id=self.args.model_id,
                  best_model_path=os.path.join(self.args.best_model_path, f"{self.args.model_id}_supervised.pth"),
                  test_loader=self.test_loader,
                  criterion=self.criterion,
                  flag="sl",
                  mae=True)


    def test(self, model, model_id, best_model_path, test_loader, criterion, flag="sl", mae=False):

        # Load best model
        model.load_state_dict(torch.load(best_model_path))

        # Evaluate on test set
        stats = iter(self.evaluate(model=model, model_id=model_id, loader=test_loader, criterion=criterion, flag=flag, mae=mae))
        test_loss = next(stats)
        print(f"Test loss (best model): {test_loss}")
        if self.logger_run is not None: self.logger_run[f"{flag}_test/loss"].append(test_loss)

        # MAE
        if mae:
            test_mae = next(stats)
            print(f"Test MAE (best model): {test_mae}")
            if self.logger_run is not None: self.logger_run[f"{flag}_test/mae"].append(test_mae)

        # Print & log test accuracy (optional)
        if self.args.acc:
            test_acc = next(stats)
            print(f"Test accuracy (best model): {test_acc}")
            if self.logger_run is not None: self.logger_run[f"{flag}_test/accuracy"].append(test_acc)

    def evaluate(self, model: nn.Module, model_id: str, loader: DataLoader, criterion: nn.Module, flag: str, mae=True, acc=False):
        """
            Evaluate the model return evaluation loss and/or evaluation accuracy.
        """
        print("Evaluating")
        stats = []
        n = len(loader)
        mae = nn.L1Loss()

        model.eval()
        with torch.no_grad():
            total_loss = total_mae = total_correct = 0
            start_time = time.time()

            for i, (x, y, _, _) in enumerate(loader):
                x = x.permute(0, 2, 1).float().to(self.device)
                y = y[:, -self.args.pred_len:].permute(0, 2, 1).float().to(self.device) # Get last pred_len values in y (get rid of overlap)
                output = model(x)
                total_loss += compute_loss(output, y, criterion, model_id)
                if mae:
                    total_mae += mae(output, y)

                # Timing
                if (i+1) % 100 == 0:
                    end_time = time.time()
                    print(f"Batch {i+1}/{len(loader)}: {end_time - start_time }s. Loss: {total_loss}")
                    start_time = time.time()

                # Compute accuracy (optional)
                if acc:
                    total_correct += (output.argmax(1) == y).type(torch.float).sum().item()

        # Average loss
        avg_loss = total_loss / n
        stats.append(avg_loss)

        # Average MAE (optional)
        if mae:
            avg_mae = total_mae / n
            stats.append(avg_mae)

        # Average accuracy (optional)
        if acc:
            avg_acc = total_correct / n
            stats.append(avg_acc)

        return tuple(stats)

    def count_parameters(self):
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model has {num_params} parameters.")
