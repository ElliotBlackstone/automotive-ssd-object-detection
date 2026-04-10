# GitHub Pages SSD demo

This is a static GitHub Pages version of the old FastAPI demo.

## Important constraint

GitHub Pages cannot run Python. This version therefore does **not** use `ssd_demo_app.py`
or `SSDInt8_ONNX_Pred_v2.py` directly.

It expects a **stitched ONNX model** that already includes decode + NMS in the graph.
Place that model at:

```text
./models/ssd_int8_with_pre_post.onnx
```

## Files

- `index.html` - page layout and upload UI
- `style.css` - styling
- `app.js` - browser-side preprocessing, ONNX Runtime Web inference, and box drawing

## Deploy

1. Put these files in a GitHub repo.
2. Add your stitched ONNX model under `models/ssd_int8_with_pre_post.onnx`.
3. Push to GitHub.
4. Enable GitHub Pages for the repo.

## Notes

- The browser version uses ImageNet normalization and resizes to 300x300.
- It assumes ONNX outputs something like `boxes_out`, `scores_out`, and `labels_out`.
- If your boxes are normalized to `[0, 1]`, the script scales them to the displayed image size.
- If your boxes are already in pixel coordinates, the script leaves them unchanged.
