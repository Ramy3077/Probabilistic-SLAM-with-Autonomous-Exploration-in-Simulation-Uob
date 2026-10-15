#!/bin/sh
# Assemble the static demo in demo/dist. The SLAM, exploration and control
# modules are copied from controllers/robot_controller, so the browser runs
# the same files as the Webots controller.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
CORE="$HERE/../controllers/robot_controller"
OUT="$HERE/dist"

rm -rf "$OUT"
mkdir -p "$OUT/core" "$OUT/py" "$OUT/maps"
cp "$HERE"/index.html "$HERE"/app.js "$HERE"/worker.js "$HERE"/style.css "$HERE"/favicon.svg "$HERE"/preview.webp "$OUT/"
cp "$HERE"/py/*.py "$OUT/py/"
cp "$HERE"/maps/*.json "$OUT/maps/"

# Only the modules the demo imports.
FILES=""
for f in slam/__init__.py slam/fastslam.py slam/occupancy.py slam/particles.py \
         models/motion.py \
         explore/__init__.py explore/frontiers.py explore/planner.py \
         control/path_planner.py control/waypoint_follow.py; do
  mkdir -p "$OUT/core/$(dirname "$f")"
  cp "$CORE/$f" "$OUT/core/$f"
  FILES="$FILES\"$f\","
done
printf '[%s]\n' "${FILES%,}" > "$OUT/core/manifest.json"
echo "demo built in $OUT"
