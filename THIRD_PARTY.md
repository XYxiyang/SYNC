# Attribution and component licenses

Component licenses and attribution notices apply to their respective files.

- EGNN layers derive from [vgsatorras/egnn](https://github.com/vgsatorras/egnn).
  Its MIT license is retained in `LICENSES/egnn.txt`.
- The classifier output head and utility modules derive from
  [ChemProjector](https://github.com/luost26/ChemProjector).
  Its MIT license is retained in `LICENSES/ChemProjector.txt`.
- The TargetDiff backbone derives from [TargetDiff](https://github.com/guanjq/targetdiff).
- The DecompDiff backbone derives from [ByteDance DecompDiff](https://github.com/bytedance/DecompDiff).
  The original CC BY-NC 4.0 license is retained in `legacy/decompdiff/LICENSE`.
- Molecular reconstruction includes code attributed to liGAN under GPL v2;
  its source notice is preserved in the reconstruction modules.
- Benchmark utilities use RDKit SA, SYBA, RAScore, SCScore, FSScore, GASA,
  Nonpher and AiZynthFinder. Their models, templates and stocks are distributed
  by the respective projects under their own terms.

Preserve the component licenses and source notices when redistributing this code.
