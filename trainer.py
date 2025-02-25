import jax
import flax
import optax

import jax.numpy as jnp

from flax.training.train_state import TrainState
from flax import linen as nn
from models_jax.ddpm import DDPM
from dataset import get_dataset
import numpy as np
import wandb
from tqdm.auto import tqdm
from sde_lib_jax import *
from configs.cifar10_continuous import get_config
import os
from datetime import datetime

from flax.training import checkpoints


class TrainState(TrainState):
    key: jax.Array
    ema_params: dict
    def apply_ema(self, decay: float = 0.999):
        new_ema_params = jax.tree_util.tree_map(
            lambda ema, param: decay * ema + (1 - decay) * param,
            self.ema_params,
            self.params,
        )
        return self.replace(ema_params=new_ema_params)

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.config = config
        self.batch_size = config.training.batch_size
        self.epochs = config.training.epochs
        self.learning_rate = config.optim.lr
        self.seed = config.seed

        self.model_path = "ckpt"

        self.beta_min = config.model.beta_min
        self.beta_max = config.model.beta_max
        self.N = config.model.num_scales
        self.sde = VPSDE(self.beta_min, self.beta_max, self.N)
        self.eps = 1e-5

        self.warmup = config.optim.warmup
        self.grad_clip = config.optim.grad_clip
        self.key = jax.random.PRNGKey(self.seed)
        self.initialize()
        self.init_fn()

    def initialize(self):
        self.key, subkey, dropout_key = jax.random.split(self.key, 3)
        variables = self.model.init(subkey, jnp.ones((1, 32, 32, 3)), jnp.ones((1,)), train=False)
        params = variables['params'] 

        schedule = optax.linear_schedule(
            init_value=0.,
            end_value=self.learning_rate,
            transition_steps=self.warmup)
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.grad_clip),
            optax.adam(learning_rate=schedule,
                        b1=self.config.optim.beta1,
                        eps=self.config.optim.eps)
        )
        
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            ema_params=params,
            key=dropout_key,
            tx=optimizer
        )


    def init_fn(self):
        def loss_fn(params, batch, train = False, dropout_key=None):

            key, subkey1, subkey2 = jax.random.split(self.key, 3)
            t = jax.random.uniform(subkey1, (batch.shape[0],)) * (self.sde.T - self.eps) + self.eps
            z = jax.random.normal(subkey2, batch.shape)

            mean, std = self.sde.marginal_prob(batch, t)
            x_t = mean + std[:, None, None, None] * z
            
            labels = t * 999

            if train:
                dropout_train_key = jax.random.fold_in(key=dropout_key, data=self.state.step)
                score = self.state.apply_fn({'params': params}, x_t, labels,
                                    train=train,
                                    rngs={'dropout': dropout_train_key}
                                    )
            else:
                score = self.state.apply_fn({'params': params}, x_t, labels, train=train)

            std_t = self.sde.marginal_prob(jnp.zeros_like(x_t), t)[1]
            score = -score/std_t[:, None, None, None]

            losses = jnp.square(score * std[:, None, None, None] + z)
            losses = jnp.mean(losses.reshape(losses.shape[0], -1), axis=-1)
            loss = jnp.mean(losses)
            return loss, key
        
        def step_fn(state, batch, train=False, dropout_key = None):

            if train:
                compute_loss_fn = lambda params: loss_fn(params, batch, train, dropout_key)
                grad_fn = jax.value_and_grad(compute_loss_fn, has_aux=True, allow_int=True)
                (loss, key), grads = grad_fn(state.params)
                state = state.apply_gradients(grads=grads)
                state = state.apply_ema(config.model.ema_rate)
            else:
                loss, key = loss_fn(state.ema_params, batch, train, dropout_key)
            return loss, state, key
        
        self.train_step = jax.jit(lambda state, batch: step_fn(state, batch, train=True, dropout_key=state.key))
        self.valid_step = jax.jit(lambda state, batch: step_fn(state, batch, train=False, dropout_key=None))

    def train_model(self, epoch):
        train_losses = []
        for x_minibatch, _ in tqdm(self.train_loader, desc='training loop'):
            x_minibatch = np.transpose(x_minibatch, (0, 2, 3, 1))
            loss, state, key = self.train_step(self.state, x_minibatch)
            self.state = state
            self.key = key
            train_losses.append(loss)
        train_loss = np.mean(train_losses)
        wandb.log({'Training loss': train_loss}, step=epoch)
        print("Training loss: {} at epoch {}".format(train_loss, epoch))

        valid_losses = []
        for x_minibatch, _ in self.val_loader:
            x_minibatch = np.transpose(x_minibatch, (0, 2, 3, 1))
            loss, _, key = self.valid_step(self.state, x_minibatch)
            self.key = key
            valid_losses.append(loss)
        valid_loss = np.mean(valid_losses)
        wandb.log({'Validation loss': valid_loss}, step=epoch)
        print("Validation loss: {} at epoch {}".format(valid_loss, epoch))
        return valid_loss


    def fit(self):
        print("Start Training...")
        best_loss = np.inf
        for epoch in range(self.epochs):
            valid_loss = self.train_model(epoch)
            if epoch % 50 == 0:
                # best_loss = valid_loss
                print("Saving model...")
                now = datetime.now()
                date_time = now.strftime("%Y%m%d_%H%M%S")
                ckpt_dir = os.path.abspath(f"ckpt/sde/cifar/{date_time}_{str(epoch)}.pth")
                checkpoints.save_checkpoint(ckpt_dir, self.state, epoch)
        print("Finished Training!")

    
if __name__ == "__main__":

    config = get_config()
    print(config)
    wandb.init(project='sde')

    train_loader, val_loader = get_dataset(config)
    model = DDPM(config)

    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.fit()