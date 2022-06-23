import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from lightning_modules.lightning_cifar10 import LitCIFAR10Model
from lightning_modules.lightning_cifar100 import LitCIFAR100Model
from lightning_modules.lightning_stl10 import LitSTLModel
from lightning_modules.lightning_svhn import LitSVHNModel


@hydra.main(config_path="config", config_name="base")
def main(cfg : DictConfig) -> None:
    print(f"[Hydra Config]:\n{OmegaConf.to_yaml(cfg)}")
    if cfg.train.dataset == 'CIFAR10':
        model = LitCIFAR10Model(model_name=cfg.train.model_name, 
                                batch_size=cfg.train.batch_size,
                                num_workers=cfg.train.num_workers,
                                learning_rate=cfg.train.learning_rate,
                                weight_decay=cfg.train.weight_decay,
                                momentum=cfg.train.momentum,
                                adversarial_poison_path=cfg.train.adversarial_poison_path,
                                unlearnable_poison_path=cfg.train.unlearnable_poison_path,
                                base_dataset_path=cfg.train.dataset_path,
                                augmentations_key=cfg.train.augmentations_key)
    elif cfg.train.dataset == 'CIFAR100':
        model = LitCIFAR100Model(model_name=cfg.train.model_name, 
                                 batch_size=cfg.train.batch_size,
                                 num_workers=cfg.train.num_workers,
                                 learning_rate=cfg.train.learning_rate,
                                 weight_decay=cfg.train.weight_decay,
                                 momentum=cfg.train.momentum,
                                 adversarial_poison_path=cfg.train.adversarial_poison_path,
                                 unlearnable_poison_path=cfg.train.unlearnable_poison_path,
                                 base_dataset_path=cfg.train.dataset_path,
                                 augmentations_key=cfg.train.augmentations_key)
    elif cfg.train.dataset == 'STL10':
        model = LitSTLModel(model_name=cfg.train.model_name, 
                            batch_size=cfg.train.batch_size,
                            num_workers=cfg.train.num_workers,
                            learning_rate=cfg.train.learning_rate,
                            weight_decay=cfg.train.weight_decay,
                            momentum=cfg.train.momentum,
                            adversarial_poison_path=cfg.train.adversarial_poison_path,
                            unlearnable_poison_path=cfg.train.unlearnable_poison_path,
                            base_dataset_path=cfg.train.dataset_path,
                            augmentations_key=cfg.train.augmentations_key)
    elif cfg.train.dataset == 'SVHN':
        model = LitSVHNModel(model_name=cfg.train.model_name, 
                            batch_size=cfg.train.batch_size,
                            num_workers=cfg.train.num_workers,
                            learning_rate=cfg.train.learning_rate,
                            weight_decay=cfg.train.weight_decay,
                            momentum=cfg.train.momentum,
                            adversarial_poison_path=cfg.train.adversarial_poison_path,
                            unlearnable_poison_path=cfg.train.unlearnable_poison_path,
                            base_dataset_path=cfg.train.dataset_path,
                            augmentations_key=cfg.train.augmentations_key)
    else:
        raise ValueError(f"Dataset {cfg.train.dataset} not supported.")

    wandblogger = WandbLogger(project=cfg.misc.project_name,
                              name=cfg.misc.run_name, 
                              log_model=cfg.misc.log_model,
                              save_dir=cfg.misc.wandb_save_dir)
    checkpoint_callback = ModelCheckpoint(monitor="epoch", 
                                          mode="max",
                                          save_top_k=-1,
                                          filename='{epoch}',
                                          dirpath=cfg.misc.dirpath)

    trainer = Trainer(gpus=-1,
                      strategy="ddp", 
                      auto_scale_batch_size=cfg.misc.use_auto_scale_batch_size, 
                      max_epochs=cfg.train.epochs, 
                      logger=wandblogger,
                      log_every_n_steps=cfg.misc.log_every_n_steps,
                      enable_progress_bar=cfg.misc.enable_progress_bar,
                      enable_checkpointing=cfg.misc.enable_checkpointing,
                      callbacks=([checkpoint_callback] if cfg.misc.enable_checkpointing else []))

    if cfg.misc.use_auto_scale_batch_size:
        trainer.tune(model)

    trainer.fit(model)
    trainer.test(model)

if __name__ == "__main__":
    main()