# Visual review of fixed real test examples

Each overlay is the first numerically selected frame in its scene. Green is BOP visible-box ground truth; red, blue, and yellow are the highest-scoring boxes for BOP IDs 2, 3, and 5. They are displayed even when their raw SVM score is below zero. The images are 640×480 real BOP YCB-V RGB; these are visual checks of detector output, not rendered Isaac images.

| Scene | Visible target(s) | Observation and judgment |
| --- | --- | --- |
| 49 | Sugar box (3) | The top blue box covers only the upper portion of the yellow sugar box. It is too small for a true positive at IoU 0.5. |
| 52 | Mustard bottle (5) | The top yellow box misses the full mustard silhouette. This accords with zero AP50 for ID 5. |
| 53 | None | The top boxes mark other containers. This is an unambiguous target-class false-positive example. |
| 54 | Cracker box (2), sugar box (3) | Some boxes overlap objects, but the highest boxes do not consistently contain the full visible extent. |
| 55 | Sugar box (3) | The top box does not reliably localize the sugar-box body. |
| 56 | None | Predicted target boxes appear on non-target objects and background. |
| 58 | Sugar box (3) | A correctly localized box occurs in the ranked list for some frames, but the highest box on this fixed example remains imperfect. |
| 59 | Cracker box (2) | Partial overlap is visible, with distracting responses elsewhere. |

All eight files in `overlays/` were opened and checked at native 640×480 resolution. These observations support the low measured AP and the conclusion that this HOG/SVM baseline is weak. They do not establish a cause for the weakness: limited synthetic samples, box scale, texture mismatch, occlusion, and classifier capacity are all plausible and were not isolated here.
