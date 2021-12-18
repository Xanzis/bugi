# BUGI
Finite Element toolset from scratch in pure rust. Name is short for 'buckled girder': that which lets down the grinding span.

The program currently supports 2D Chew meshing, static linear elastic simulation, and linear modal analysis. To get started, compile the source and run 

```
bugi --help
```

## Example Commands

```
bugi mesh example_files/cantilever.bbnd -s 0.2
bugi linear out.bmsh -s cholesky -n vonmises
```
Generates a mesh of a provided cantilevered beam geometry, then calculates a linear elastic solution for the loaded beam. Saves a plot of the beam which visualizes the computed von Mises criterion values throughout the beam.

```
bugi mesh example_files/hollow.bbnd -s 0.1
bugi modal out.bmsh 2
```
Finds the first two resonant modes of a hollow square using the determinant search method.
