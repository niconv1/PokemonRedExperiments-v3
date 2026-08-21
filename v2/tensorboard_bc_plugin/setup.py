from setuptools import setup

setup(
    name="pokemonred-tensorboard-bc",
    version="0.1.0",
    packages=["pokemonred_tensorboard_bc"],
    entry_points={
        "tensorboard_plugins": [
            "pokemonred_bc = pokemonred_tensorboard_bc.plugin:BCPlugin",
        ],
    },
)
