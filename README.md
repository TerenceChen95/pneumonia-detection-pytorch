# pneumonia-detection-pytorch
a pytorch implementation of kagge pneumonia competition
## DataSet
Data are installed from kaggle competition 'RSNA Pneumonia Detection Challenge'
## Main Idea
- 3-class classification
  - lung opacity
  - normal
  - not normal / no lung opacity
- trian with densenet121 
  - fine-tuning classifier
- Grad_CAM unsupervised learning
  - improve explainability
  - thresholding heatmap to output bbox
  
![]('./pictures/predict_img.png')
