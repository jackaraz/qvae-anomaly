#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quantum Variational Autoencoder for anomaly detection"""

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
from pennylane.operation import AnyWires, Operation
from sklearn.datasets import make_circles, make_moons, make_s_curve
from sklearn.model_selection import train_test_split

jax.config.update("jax_enable_x64", True)

# pylint: disable=W0621,C0103,W1514,C0200,R0913

_rot = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}


class Layer(Operation):
    """Based on StronglyEntangling Layer"""

    num_wires = AnyWires
    grad_method = None

    def __init__(
        self,
        inpts,
        weights,
        wires,
        reupload: bool = True,
        rot: List[Operation] = None,
        alternate_embedding: bool = False,
    ):
        shape = qml.math.shape(weights)
        range_len = shape[0]
        if len(wires) > 1:
            # tile ranges with iterations of range(1, n_wires)
            ranges = tuple((l % (len(wires) - 1)) + 1 for l in range(range_len))
        else:
            ranges = (0,) * range_len

        if rot is None:
            rot = [qml.RY]

        self._hyperparameters = {
            "ranges": ranges,
            "inputs": inpts,
            "reupload": reupload,
            "rot": rot,
            "alternate_embedding": alternate_embedding,
        }

        super().__init__(weights, wires=wires, id=None)

    @staticmethod
    def compute_decomposition(
        weights, wires, ranges, inputs, reupload, rot, alternate_embedding
    ):  # pylint: disable=arguments-differ, too-many-arguments, too-many-locals

        weight_shape = qml.math.shape(weights)
        n_layers = weight_shape[0]
        wires = qml.wires.Wires(wires)

        index = sorted(list(range(len(inputs))) * (len(wires) // len(inputs)))
        rotemb = [qml.RY, qml.RX if alternate_embedding else qml.RY]
        embeding = [
            rotemb[idx % 2](inputs[index[idx]], wires=wires[idx])
            for idx in range(len(wires))
        ]

        op_list = []
        if not reupload:
            op_list += embeding

        # nlayer = 0
        for l in range(n_layers):
            for i in range(len(wires)):
                op_list += [
                    rot[jdx](weights[..., l, i, jdx], wires=wires[i])
                    for jdx in range(len(rot))
                ]

            if len(wires) > 1:
                for i in range(len(wires)):
                    act_on = wires.subset([i, i + ranges[l]], periodic_boundary=True)
                    op_list.append(qml.CNOT(wires=act_on))

            if reupload and l < n_layers - 1:
                op_list += embeding

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
    parallel_embedding: int = 1,
    alternate_embedding: bool = False,
) -> Tuple[Callable, List[int]]:
    """
    Construct qVAE circuit

    Args:
        ndata (``int``): data dimensionality
        nref (``int``): number of reference wires
        nlayers (``int``, default ``1``): number of layers. Defaults to 1.
        reupload (``bool``, default ``True``):  use data-reuploading. Defaults to True.
        rotseq (``List[Text]``, default ``None``): rotation sequence for trainable layer.
            Choices are `"X"`, `"Y"` and `"Z"`.
        parallel_embedding (``int``, default ``1``): Embed data on multiple qubits
        alternate_embedding (``bool``, default ``False``): alternate the angle embedding rotation
            between `"Y"` and `"X"`.

    Returns:
        ``Tuple[Callable, List[int]]``:
        circuit and parameter shape
    """
    rotseq = [_rot["Y"]] if rotseq is None else [_rot[r.upper()] for r in rotseq]

    n_vqa_wires = ndata * parallel_embedding
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
            alternate_embedding=alternate_embedding,
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


def circle(
    samples: int, center: List[List[float]] = None, radius: List[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Circle data

    Args:
        samples (int): number of samples
        center (List[List[float, float]], optional): Center of the circles. Defaults to [[0.0, 0.0]].
        radius (List[float], optional): radius of the circles. Defaults to [np.sqrt(2 / np.pi)].

    Returns:
        `Tuple[np.ndarray, np.ndarray]`:
    """
    center = center or [[0.0, 0.0]]
    radius = radius or [np.sqrt(2.0 / np.pi)]

    x = 2 * np.random.random((samples, 2)) - 1
    y = np.zeros(samples)
    for c, r in zip(center, radius):
        y[np.linalg.norm(x - c, axis=1) < r] = 1
    return x, y


def get_data(data_source: Text, feat_dim: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retreive data

    Args:
        data_source (``Text``): data source, either a csv file or
            "circle", "circles", "moons", "s_curve"
        feat_dim (``int``, default ``-1``): limit the dimensionality of the features
            (only for csv data)

    Raises:
        ``ValueError``: Raised if the data source is unknown

    Returns:
        ``Tuple[np.ndarray, np.ndarray]``:
        training and validation data
    """
    if data_source.endswith(".csv"):
        # Standardized data
        data = pd.read_csv(data_source, delimiter=",")
        nfeat = feat_dim if feat_dim > 0 else data.shape[-1]
        X_train, X_val = train_test_split(
            data.values[:, :nfeat], test_size=0.2, shuffle=False
        )
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
        f"   * Number of training samples {len(X_train)}, "
        f"number of validation samples {len(X_val)}, "
        f"{X_train.shape[-1]}D feature space"
    )
    return X_train, X_val


def train(args):
    """Execute training routine"""

    jax.config.update("jax_platform_name", "gpu" if args.GPU else "cpu")

    X_train, X_val = get_data(data_source=args.DATAPATH, feat_dim=args.FEATDIM)

    assert (
        args.NREF < X_train.shape[-1] * args.PAREMBED
    ), "Number of reference qubits should be less than input dimensions."

    circ, shape = qvae(
        ndata=X_train.shape[-1],
        nref=args.NREF,
        nlayers=args.NLAYERS,
        reupload=args.REUPLOAD,
        rotseq=args.ROTSEQ,
        parallel_embedding=args.PAREMBED,
        alternate_embedding=args.ALTEMBED,
    )

    optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=args.ETA)
    scheduler = optax.exponential_decay(
        init_value=args.ETA,
        transition_steps=100,
        decay_rate=0.5,
        staircase=True,
        end_value=1e-4,
    )

    batch_cost, train_step = get_cost(circ, optimizer, args.LINLOSS)

    parameters = jnp.array(np.random.uniform(-np.pi, np.pi, shape))
    opt_state = optimizer.init(parameters)

    to_save, min_val_loss, best_idx = jnp.zeros(shape), np.inf, 0

    train_loss, val_loss, lr_state = [], [], []
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

            opt_state.hyperparams["learning_rate"] = scheduler(epoch + 1)
            lr_state.append(float(opt_state.hyperparams["learning_rate"]))

            pbar.set_postfix_str(
                f"train loss: {train_loss[-1]:.3e}, val loss: {val_loss[-1]:.3e}, "
                f"lr: {lr_state[-1]:.3e}"
            )

            # Save only the best model
            if val_loss[-1] <= min_val_loss:
                to_save = np.array(parameters)
                min_val_loss = val_loss[-1]
                best_idx = epoch

            pbar.update()

    with open(os.path.join(args.OUTPATH, "config.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f)
    np.savez_compressed(
        os.path.join(args.OUTPATH, "results.npz"),
        param=to_save,
        train_loss=train_loss,
        val_loss=val_loss,
        lr=lr_state,
        best_idx=best_idx,
    )
    print(f" * Output folder: {args.OUTPATH}")


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
        "--parallel-embedding",
        "-par-emb",
        type=int,
        default=1,
        help="Embed the dataset multiple times (increases number of qubits). Defaults to 1",
        dest="PAREMBED",
    )
    parameters.add_argument(
        "--alternate-embedding",
        "-alt-emb",
        action="store_true",
        default=False,
        help="Alternate embedding procedure i.e. one qubit RY one qubit RX "
        "(designed for parallel embedding)",
        dest="ALTEMBED",
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
    parameters.add_argument(
        "--feature-dimension",
        "-feat-dim",
        type=int,
        default=-1,
        help="Number of features to be included for training. Defaults to -1",
        dest="FEATDIM",
    )

    exe = parser.add_argument_group("Execution type.")
    exe.add_argument(
        "-gpu", action="store_true", default=False, help="Execute as on GPU", dest="GPU"
    )

    data = parser.add_argument_group("Options for data.")
    data.add_argument(
        "--data-path",
        "-dp",
        type=str,
        help="Data CSV file",
        dest="DATAPATH",
    )

    path = parser.add_argument_group("Options for paths.")
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

    print("<><><> Arguments <><><>")
    for key, item in vars(args).items():
        print(f"   * {key} : {item}")
    print("<><><><><><><><><><><>")

    train(args)
