# Example: add an Iron Man helmet to a Spider-Man figure

Instruction: `Spider-Man with Iron Man's helmet`.
The target image supplies the requested edit; the instruction is descriptive.

| File | Purpose |
|---|---|
| `model.glb` | Input 3D asset |
| `source_render.png` | Source view |
| `target_edit.png` | Target view describing the edit |

After following the installation and weight layout in the root README, edit the
paths at the top of `run.sh` and run from the repository root:

```bash
bash run.sh
```

The output directory receives preprocessing PLY files, `edited.glb`,
`normalized.glb`, and the default preview `turntable.mp4`.
Input files are not modified. Existing weights and server paths can be supplied
through the variables documented in `run.sh`.

These inputs were selected from the existing Edit3D-Bench/GSO experiment assets.
