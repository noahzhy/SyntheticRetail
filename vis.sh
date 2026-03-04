export frame_idx=0017

python3.12 utils/visual.py \
    --labels labels/sequence_20260206_162556/frame_${frame_idx}_20260206_162556.json \
    --image labels/sequence_20260206_162556/frame_${frame_idx}_20260206_162556.jpg \
    --save preview.png \
    --no-show --hide-occluded

echo "preview.png ready."
