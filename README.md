# DisPerSe_Wrap

Python wrapper for [DisPerSE](https://www2.iap.fr/users/sousbie/web/html/indexd41d.html) –
a tool for identifying topological features (filaments, walls, voids) in cosmological simulations and surveys.

## Quick start

```bash
pip install -e python
export DISPERSE_BIN=/path/to/disperse/bin
disperse-wrapper run --points subhalos.npy --boxsize 100.0 --periodic --out ./output
```

See [python/README.md](python/README.md) for full documentation.
