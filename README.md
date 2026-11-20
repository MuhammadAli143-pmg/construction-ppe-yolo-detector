# Construction PPE Object Detector (YOLO)

A Streamlit application that detects construction/PPE-related objects in images with a fine-tuned Ultralytics YOLO model. The training notebook is included but intentionally **not run**.

## Dataset
Use a public Construction/PPE detection dataset from Roboflow Universe with at least two classes and download/export it in YOLOv8 format. Dataset versions/class lists vary, so confirm the classes and license on the chosen project page.

## Train (Google Colab)
1. Open `training/PPE_Object_Detection_YOLO_Training.ipynb` in Google Colab.
2. Enable GPU runtime.
3. Follow the Roboflow dataset download cell and enter workspace/project/version/API key. Do not expose your API key in a public repository.
4. Run cells in order. The notebook trains for 20 epochs and reports validation mAP50 and mAP50-95.
5. Copy the resulting `best.pt` into `models/best.pt` in this repo.

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud
1. Push the repository to GitHub.
2. Create a Streamlit Community Cloud app, select this repo and `streamlit_app.py`.
3. Ensure `models/best.pt` is committed (Git LFS may be needed if the file is large; Cloud deployment may have size limits).
4. Deploy, test image upload/confidence slider, and capture two screenshots showing detections.

## Submission checklist
- [ ] Public deployed app link added below
- [ ] GitHub repository link added below
- [ ] Notebook run and showing epoch history and final mAP50
- [ ] Two screenshots saved in `screenshots/`

### Links
- Public app: **ADD AFTER DEPLOYMENT**
- GitHub repository: **ADD AFTER PUSHING**
