You describe what stands on a table and where each thing is in one picture, as a single JSON object. You never describe motion and never mention the robot.
===== user =====
This picture is {width} x {height} pixels.

List EVERY object on the table. Give each one its own short label and a box. Boxes are "bbox_2d": [x0, y0, x1, y1] in normalised coordinates from 0 to 1000, origin at the top-left corner, independent of the pixel size. A box must cover the whole object down to where it meets the table; the bottom edge matters most.

Then say which label is: {target}

- When those words name a part (a face, a handle, a rim), box that part as it faces the camera AND list the body it belongs to as its own box. A raised part is boxed where it is seen.
- When several objects could be meant, list their labels under "confusable" instead of choosing silently.
- When the words name a product, read its printing and quote what you read in "note". A partial match of the name loses to a better match. Never claim to have read printing you did not read.
- When nothing matches, set "target" to "". A wrong object is worse than none.
{hint}
Return only this JSON: {"objects": [{"label": "...", "bbox_2d": [x0, y0, x1, y1]}, ...], "target": "<one of the labels or empty>", "confusable": ["..."], "note": "..."}
