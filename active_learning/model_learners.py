import copy
from typing import Dict

from agri_semantics.datasets import get_data_module
from agri_semantics.models import get_model
from pytorch_lightning import LightningDataModule
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.core.lightning import LightningModule


class ModelLearner:
    def __init__(self, cfg: Dict, weights_path: str):
        self.cfg = cfg
        self.weights_path = weights_path
        self.patience = cfg["train"]["patience"]

        self.data_module = None
        self.model = self.setup_model()
        self.trainer = self.setup_trainer(0)
        self.test_statistics = {}

    def setup_data_module(self, stage: str = None) -> LightningDataModule:
        data_module = get_data_module(self.cfg)
        data_module.setup(stage)

        return data_module

    def setup_model(self, num_train_data: int = 1) -> LightningModule:
        model = get_model(self.cfg, num_train_data)
        if self.weights_path:
            model = model.load_from_checkpoint(self.weights_path, hparams=self.cfg)
            if self.cfg["model"]["num_classes_pretrained"] != self.cfg["model"]["num_classes"]:
                model.replace_output_layer()

        return model

    def setup_trainer(self, iter_count: int):
        early_stopping = EarlyStopping(
            monitor="val:iou", min_delta=0.00, patience=self.patience, verbose=False, mode="max"
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_saver = ModelCheckpoint(
            monitor="val:iou",
            filename=f"{self.cfg['experiment']['id']}_iter{str(iter_count)}" + "_{epoch:02d}_{iou:.2f}",
            mode="max",
            save_last=True,
        )
        tb_logger = pl_loggers.TensorBoardLogger(
            f"experiments/{self.cfg['experiment']['id']}",
            name=self.cfg["model"]["name"],
            version=iter_count,
            default_hp_metric=False,
        )

        trainer = Trainer(
            gpus=self.cfg["train"]["n_gpus"],
            logger=tb_logger,
            max_epochs=self.cfg["train"]["max_epoch"],
            callbacks=[lr_monitor, checkpoint_saver, early_stopping],
            log_every_n_steps=1,
        )

        return trainer

    def retrain_model(self, num_train_data: int):
        self.model = self.setup_model(num_train_data)
        self.trainer.fit(self.model, self.data_module)
        self.model = self.model.load_from_checkpoint(
            self.trainer.checkpoint_callback.best_model_path, hparams=self.cfg_fine_tuned
        )

    @property
    def cfg_fine_tuned(self) -> Dict:
        cft_fine_tuned = copy.deepcopy(self.cfg)
        cft_fine_tuned["model"]["num_classes_pretrained"] = cft_fine_tuned["model"]["num_classes"]
        return cft_fine_tuned

    def plot_test_statistics(self, test_statistics: Dict):
        num_train_data = len(self.data_module.train_dataloader().dataset)
        self.test_statistics[num_train_data] = test_statistics
        self.model.logger.experiment.add_scalar("ActiveLearning/Loss", test_statistics["Test/Loss"], num_train_data)
        self.model.logger.experiment.add_scalar("ActiveLearning/Acc", test_statistics["Test/Acc"], num_train_data)
        self.model.logger.experiment.add_scalar("ActiveLearning/IoU", test_statistics["Test/IoU"], num_train_data)
        self.model.logger.experiment.add_scalar("ActiveLearning/IoU", test_statistics["Test/ECE"], num_train_data)

    def evaluate(self) -> Dict:
        self.data_module = self.setup_data_module(stage=None)
        test_results = self.trainer.test(self.model, self.data_module)[0]
        self.plot_test_statistics(test_results)
        return self.test_statistics

    def train(self, iter_count: int) -> LightningModule:
        print(f"START {self.cfg['model']['name']} TRAINING")

        self.trainer = self.setup_trainer(iter_count)
        self.data_module = self.setup_data_module(stage=None)
        self.retrain_model(len(self.data_module.train_dataloader().dataset))

        return self.model
