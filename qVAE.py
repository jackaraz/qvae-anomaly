#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""qVAE for a given dataset"""

import argparse
import os
from datetime import datetime
from typing import Callable, Iterator, List, Text, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pennylane as qml
import tqdm
import yaml
from flax.training.early_stopping import EarlyStopping
from pennylane.operation import AnyWires, Operation
from sklearn.datasets import make_circles, make_moons, make_s_curve
from sklearn.model_selection import train_test_split

jax.config.update("jax_enable_x64", True)

# pylint: disable=W0621,C0103,W1514

_rot = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}


class Layer(Operation):
    """Based on StronglyEntangling Layer"""

    num_wires = AnyWires
    grad_method = None

    def __init__(
        self, inpts, weights, wires, reupload: bool = True, rot: List[Operation] = None
    ):
        shape = qml.math.shape(weights)[-3:]
        assert len(inpts) == len(
            wires
        ), f"wrong dimensionality {len(inpts)} != {len(wires)}"

        if len(wires) > 1:
            # tile ranges with iterations of range(1, n_wires)
            ranges = tuple((l % (len(wires) - 1)) + 1 for l in range(shape[0]))
        else:
            ranges = (0,) * shape[0]

        if rot is None:
            rot = [qml.RY]

        self._hyperparameters = {
            "ranges": ranges,
            "inputs": inpts,
            "reupload": reupload,
            "rot": rot,
        }

        super().__init__(weights, wires=wires, id=None)

    @staticmethod
    def compute_decomposition(
        weights, wires, ranges, inputs, reupload, rot
    ):  # pylint: disable=arguments-differ, too-many-arguments

        n_layers = qml.math.shape(weights)[0]
        wires = qml.wires.Wires(wires)
        embedding = [qml.RY(inputs[i], wires=wires[i]) for i in range(len(wires))]
        op_list = []
        if not reupload:
            op_list += embedding

        for l in range(n_layers):
            for i in range(len(wires)):  # pylint: disable=consider-using-enumerate
                op_list += [
                    rot[jdx](weights[..., l, i, jdx], wires=wires[i])
                    for jdx in range(len(rot))
                ]

            if len(wires) > 1:
                for i in range(len(wires)):
                    act_on = wires.subset([i, i + ranges[l]], periodic_boundary=True)
                    op_list.append(qml.CNOT(wires=act_on))

            if reupload and l < n_layers - 1:
                op_list += embedding

        return op_list

    @staticmethod
    def shape(n_layers: int, n_wires: int, nrot: int) -> Tuple[int, int, int]:
        """Shape of the input

        Args:
            n_layers (int): number of layers
            n_wires (int): number of wires
            nrot (int): number of rotation gates

        Returns:
            Tuple[int, int, int]: (nlayer, nwire, nrot)
        """
        return n_layers, n_wires, nrot


def qvae(
    ndata: int,
    nref: int,
    nlayers: int = 1,
    reupload: bool = True,
    rotseq: List[Text] = None,
) -> Tuple[Callable, List[int]]:
    """
    Construct qVAE circuit

    Args:
        ndata (``int``): data dimensionality
        nref (``int``): number of reference wires
        nlayers (``int``, optional): number of layers. Defaults to 1.
        reupload (``bool``, optional): use data-reuploading. Defaults to True.

    Returns:
        Tuple[Callable, List[int]]: circuit and parameter shape
    """
    rotseq = [_rot["Y"]] if rotseq is None else [_rot[r.upper()] for r in rotseq]

    n_vqa_wires = ndata
    n_reference = nref

    numb_all_wires = n_vqa_wires + n_reference + 1

    shape = list(Layer.shape(nlayers, n_vqa_wires, len(rotseq)))

    @qml.qnode(qml.device("default.qubit.jax", wires=numb_all_wires), interface="jax")
    def qvae_circuit(inpt: jnp.array, param: jnp.array) -> jnp.array:
        Layer(
            inpts=inpt,
            weights=param,
            wires=range(n_vqa_wires),
            reupload=reupload,
            rot=rotseq,
        )

        # SWAP test to measure fidelity
        qml.Hadamard(wires=numb_all_wires - 1)
        for ref_wire, trash_wire in zip(
            range(n_vqa_wires - n_reference, n_vqa_wires),
            range(n_vqa_wires, numb_all_wires - 1),
        ):
            qml.CSWAP(wires=[numb_all_wires - 1, ref_wire, trash_wire])
        qml.Hadamard(wires=numb_all_wires - 1)
        return qml.expval(op=qml.PauliZ(wires=numb_all_wires - 1))

    return qvae_circuit, shape


def batch_split(
    data: np.array, batch_size: int, shuffle: bool = True
) -> Iterator[jnp.array]:
    """Split data into batches

    Args:
        data (np.array): data to be splitted
        batch_size (int): size of each batch
        shuffle (bool, optional): Should the batch be shuffled. Defaults to True.

    Yields:
        Iterator[jnp.array]: batched data
    """
    indices = np.arange(len(data))
    if shuffle:
        np.random.shuffle(indices)
    batches = np.array_split(indices, len(indices) // batch_size)
    if shuffle:
        np.random.shuffle(batches)
    return (jnp.array(data[batch, :]) for batch in batches)


def get_cost(circuit, optimizer, linear_loss: bool = False):
    """Construct the cost function"""

    if linear_loss:

        @jax.jit
        def batch_cost(data, param):
            return jnp.mean(
                1.0 - (jax.vmap(lambda dat: circuit(dat, param), in_axes=0)(data))
            )

    else:

        @jax.jit
        def batch_cost(data, param):
            return jnp.mean(
                -jnp.log(jax.vmap(lambda dat: circuit(dat, param), in_axes=0)(data))
            )

    value_and_grad = jax.value_and_grad(batch_cost, argnums=1)

    @jax.jit
    def train_step(batch: jnp.array, pars: jnp.array, opt_state):
        loss, grad = value_and_grad(batch, pars)
        updates, opt_state = optimizer.update(grad, opt_state, value=loss)
        pars = optax.apply_updates(pars, updates)
        return loss, pars, opt_state

    return batch_cost, train_step


class ReduceLROnPlateau:
    """
    Reduce learning rate on plateau

    Parameters
    ----------
    check_every : int, optional
        check every such epoch. The default is 20.
    min_lr : float, optional
        Min value that lr can take. The default is 1e-5.
    scale : float, optional
        LR scale factor. The default is 0.5.
    min_improvement_rate : float, optional
        Minimum amount of improvement which does not require lr update.
        The default is 0.01.
    store_lr : TYPE, optional
        Store lr updates. The default is False.
    """

    def __init__(
        self,
        check_every: int = 20,
        min_lr: float = 1e-5,
        scale: float = 0.5,
        min_improvement_rate: float = 0.01,
        store_lr=False,
    ):
        self.min_lr = min_lr
        self.scale = scale
        self.check_every = check_every
        self.min_improvement_rate = min_improvement_rate
        self.min_loss = 1e99
        self.store_lr = store_lr
        self.lr = []

    def __call__(self, opt_state, epoch: int, losses: List[float]):
        current_lr = opt_state.hyperparams["learning_rate"]
        if self.store_lr:
            self.lr.append(float(current_lr))

        if current_lr > self.min_lr and epoch % self.check_every == 0:

            min_loss = min(losses[-self.check_every :])
            if abs(min_loss - self.min_loss) >= self.min_improvement_rate:
                self.min_loss = min_loss
                opt_state.hyperparams["learning_rate"] = jnp.clip(
                    current_lr * self.scale, self.min_lr, None
                )
        return opt_state


def circle(
    samples: int,
    center: Tuple[List[float, float]] = ([0.0, 0.0],),
    radius: Tuple[float] = (np.sqrt(2 / np.pi),),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Circle data

    Args:
        samples (int): number of samples
        center (Tuple[List[float, float]], optional): Center of the circles. Defaults to [[0.0, 0.0]].
        radius (Tuple[float], optional): radius of the circles. Defaults to [np.sqrt(2 / np.pi)].

    Returns:
        `Tuple[np.ndarray, np.ndarray]`:
    """
    x = 2 * np.random.random((samples, 2)) - 1
    y = np.zeros(samples)
    for c, r in zip(center, radius):
        y[np.linalg.norm(x - c, axis=1) < r] = 1
    return x, y


def get_data(data_source: Text):
    """Retreive data"""
    if "csv" in data_source:
        # Standardized data
        data = pd.read_csv(data_source, delimiter=",")
        X_train, X_val = train_test_split(data.values, test_size=0.2, shuffle=False)
        X_train, _ = train_test_split(X_train, test_size=0.01, shuffle=False)
    elif data_source == "circle":
        Xdata, ydata = circle(100000, center=[(-0.45, -0.45)])
        X_train, X_val = train_test_split(Xdata[ydata == 1], test_size=0.2, shuffle=True)
    elif data_source == "circles":
        Xdata, ydata = make_circles(100000, factor=0.1, noise=0.2)
        X_train, X_val = train_test_split(Xdata[ydata == 0], test_size=0.2, shuffle=True)
    elif data_source == "moons":
        Xdata, ydata = make_moons(100000, noise=0.1)
        X_train, X_val = train_test_split(Xdata[ydata == 0], test_size=0.2, shuffle=True)
    elif data_source == "s_curve":
        Xdata = make_s_curve(10000, noise=0.15, random_state=0)[0][:, [0, 2]]
        X_train, X_val = train_test_split(Xdata, test_size=0.2, shuffle=True)
    else:
        raise ValueError(f"Unkown data source: {data_source}")

    print(
        f"   * Number of training samples {len(X_train)}, number of validation samples {len(X_val)}"
    )
    return X_train, X_val


def train(args):
    """Execute training routine"""

    jax.config.update("jax_platform_name", "gpu" if args.GPU else "cpu")

    X_train, X_val = get_data(args.DATAPATH)

    assert (
        args.NREF < X_train.shape[-1]
    ), "Number of reference qubits should be less than input dimensions."

    circ, shape = qvae(
        X_train.shape[-1],
        args.NREF,
        args.NLAYERS,
        reupload=args.REUPLOAD,
        rotseq=args.ROTSEQ,
    )

    optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=args.ETA)
    reduce_on_plateau = ReduceLROnPlateau(check_every=25, min_lr=1e-4, store_lr=True)

    batch_cost, train_step = get_cost(circ, optimizer, args.LINLOSS)

    early_stop = EarlyStopping(
        min_delta=args.MINDELTA, patience=args.PATIENCE, patience_count=30
    )

    parameters = jnp.array(np.random.uniform(-np.pi, np.pi, shape))
    opt_state = optimizer.init(parameters)

    train_loss = []
    val_loss = []
    with tqdm.tqdm(
        total=args.EPOCHS, unit="Epoch", bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"
    ) as pbar:

        for epoch in range(args.EPOCHS):
            batch_loss = []
            for batch in batch_split(X_train, args.BATCH):
                loss, parameters, opt_state = train_step(batch, parameters, opt_state)
                batch_loss.append(float(loss))
            train_loss.append(np.mean(batch_loss))

            # Validation:
            val_loss.append(
                np.mean(
                    [
                        float(batch_cost(batch, parameters))
                        for batch in batch_split(X_val, args.BATCH)
                    ]
                )
            )

            opt_state = reduce_on_plateau(opt_state, epoch + 1, np.array(val_loss))

            pbar.set_postfix_str(
                f"train loss: {train_loss[-1]:.3e}, val loss: {val_loss[-1]:.3e}, lr: {reduce_on_plateau.lr[-1]:.3e}"
            )

            # early_stop = early_stop.update(val_loss[-1])
            # if early_stop.should_stop:
            #     print(f"Met early stopping criteria, breaking at epoch {epoch}")
            #     break

            pbar.update()

    with open(os.path.join(args.OUTPATH, "config.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f)
    np.savez_compressed(
        os.path.join(args.OUTPATH, "results.npz"),
        param=np.array(parameters),
        train_loss=train_loss,
        val_loss=val_loss,
        lr=reduce_on_plateau.lr,
    )
    print(f" * Output folder: {args.OUTPATH}")


# def test(args):
#     data = pd.read_csv(args.DATAPATH, delimiter=",", index_col=0)
#     data = data[
#         (data.lep1pt < 1000.0)
#         & (data.lep2pt < 900.0)
#         & (data.theta_ll < np.pi)
#         & (data.b1pt < 1000.0)
#         & (data.b2pt < 900.0)
#         & (data.theta_bb < np.pi)
#         & (data.MET < 1000.0)
#     ]
#     scaler = MinMaxScaler(feature_range=(-np.pi, np.pi)).fit(data.values)
#     del data

#     test_data = pd.read_csv(args.TESTDATAPATH, delimiter=",", index_col=0)
#     test_data = test_data[
#         (test_data.lep1pt < 1000.0)
#         & (test_data.lep2pt < 900.0)
#         & (test_data.theta_ll < np.pi)
#         & (test_data.b1pt < 1000.0)
#         & (test_data.b2pt < 900.0)
#         & (test_data.theta_bb < np.pi)
#         & (test_data.MET < 1000.0)
#     ]
#     X_test = scaler.transform(test_data.values)
#     del test_data

#     with open(os.path.join(args.RESPATH, "config.yaml"), "r") as f:
#         config = yaml.safe_load(f)

#     opt_res = np.load(os.path.join(args.RESPATH, "results.npz"))
#     circuit, shape = qvae(
#         X_test.shape[1], config["NREF"], config["NLAYERS"], config["REUPLOAD"]
#     )

#     @jax.jit
#     def batch_fid(data, param):
#         return jax.vmap(lambda dat: circuit(dat, param), in_axes=0)(data)

#     sig = batch_fid(X_test, opt_res["param"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test or optimise qVAE for anomaly detection"
    )
    parameters = parser.add_argument_group("Set parameters of the model.")
    parameters.add_argument(
        "-nref",
        type=int,
        default=3,
        help="Number of reference qubits, default 3.",
        dest="NREF",
    )
    parameters.add_argument(
        "-nlayers",
        type=int,
        default=1,
        help="Number of ansätz layers, default 1.",
        dest="NLAYERS",
    )
    parameters.add_argument(
        "--nepochs",
        "-ne",
        type=int,
        default=300,
        help="Number of epochs, default 500.",
        dest="EPOCHS",
    )
    parameters.add_argument(
        "--min-delta",
        "-md",
        type=float,
        default=1e-4,
        help="Minimum delta for early stopping, default 1e-4.",
        dest="MINDELTA",
    )
    parameters.add_argument(
        "--patientce",
        "-pat",
        type=int,
        default=150,
        help="Patience for early stopping, default 100.",
        dest="PATIENCE",
    )
    parameters.add_argument(
        "--batch-size",
        "-bs",
        type=int,
        default=100,
        help="Batch size, default 100.",
        dest="BATCH",
    )
    parameters.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=0.1,
        help="Learning rate, default 0.1.",
        dest="ETA",
    )
    parameters.add_argument(
        "--reupload",
        action="store_true",
        default=False,
        help="Execute data-reuploading circuit.",
        dest="REUPLOAD",
    )
    parameters.add_argument(
        "--linear-loss",
        "-linl",
        action="store_true",
        default=False,
        help="Use 1-Fidelity loss instead of -logFid",
        dest="LINLOSS",
    )
    parameters.add_argument(
        "--rotation-sequence",
        "-rot-seq",
        default=["Y"],
        nargs="+",
        type=str,
        help="Rotation sequence, default Y",
        dest="ROTSEQ",
    )

    exe = parser.add_argument_group("Execution type.")
    exe.add_argument(
        "-test",
        action="store_true",
        default=False,
        help="Execute as testing routine",
        dest="TEST",
    )
    exe.add_argument(
        "-gpu", action="store_true", default=False, help="Execute as on GPU", dest="GPU"
    )
    exe.add_argument(
        "--results-path",
        "-rp",
        type=str,
        help="Path to the model results to be tested.",
        dest="RESPATH",
    )

    data = parser.add_argument_group("Options for data.")
    data.add_argument(
        "--data-path",
        "-dp",
        type=str,
        help="Data CSV file",
        dest="DATAPATH",
    )
    data.add_argument(
        "--test-data-path",
        "-tdp",
        type=str,
        help="Data CSV file for test set",
        dest="TESTDATAPATH",
    )

    path = parser.add_argument_group("Options for paths.")
    path.add_argument(
        "--model-path",
        "-mp",
        type=str,
        help="Model configuration path, only for testing",
        dest="MODELPATH",
    )
    path.add_argument(
        "--out-path",
        "-op",
        type=str,
        help="Output path, detault `./results_" + datetime.now().strftime("%b%d") + "`",
        dest="OUTPATH",
        default="./results_" + datetime.now().strftime("%b%d"),
    )
    path.add_argument(
        "--out-name",
        "-on",
        type=str,
        help="Output name, default " + datetime.now().strftime("%b%d_%I-%M-%S%p"),
        dest="OUTNAME",
        default=datetime.now().strftime("%b%d_%I-%M-%S%p"),
    )

    args = parser.parse_args()

    if not os.path.isdir(args.OUTPATH):
        os.mkdir(args.OUTPATH)

    args.OUTPATH = os.path.join(args.OUTPATH, args.OUTNAME)
    if not os.path.isdir(args.OUTPATH):
        os.mkdir(args.OUTPATH)

    # if args.TEST:
    #     test(args)

    train(args)
