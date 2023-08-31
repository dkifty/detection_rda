# detection_rda
start with

```python

import sys
sys.path.append('./utils')

```
video to frame

```python

from vid2frame import v2f
v2f(folder_name, fomatting, frame)

# folder_name = Folder name that contain the videos ex) '230831' or '230831/1' -> str
# fomatting = format of video ex) 'MP4' -> str
# frame = interval that you want to slice for frames ex) 30 -> int
# ex) v2f('230831', 'MP4', 60)
```
