# Fixed-frame visual review

All nine overlays were opened at their native 640×480 size. Green is BOP visible-box ground truth; red, blue, and yellow are the highest-scoring SIFT candidate for BOP IDs 2, 3, and 5. A missing colored box means this detector made no candidate for that class on the shown frame. These are real camera images except for scene 40, which is BlenderProc synthetic.

| Scene | Observation |
| --- | --- |
| Synthetic 40 | The red cracker-box and yellow mustard-box candidates overlap their targets in the fixed example; the sugar-box candidate is absent. This agrees with the uneven class metrics. |
| Real 49 | Sugar box is visible in green, but the first chosen frame has no sugar candidate. Other frames contribute to scene AP50 0.378. |
| Real 52 | Mustard bottle is clear in green; no yellow candidate appears. All 11 mustard instances were missed. |
| Real 53 | There is no target object. A red candidate marks another can, a false positive. |
| Real 54 | The sugar-box candidate overlaps the target. The cracker-box candidate is much wider than the visible box and includes neighboring objects. |
| Real 55 | The sugar box is missed in this frame; a broad red box marks background and other objects. |
| Real 56 | There is no target object. A small red box appears on the drill, another false positive. |
| Real 58 | The sugar-box candidate covers the box and adjacent Jell-O package, showing localization imprecision. |
| Real 59 | The cracker-box candidate overlaps the target but is larger than its visible outline. |

The images support the reported class asymmetry and localization limits. They do not establish a single cause; occlusion, texture, viewpoint, and homography fit were not individually controlled.
