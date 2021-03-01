# Comparing Diabetes Risk Assessment Scores
--Can we create a better predictive system for T2DM risk via machine learning or weighting factors as compared to the classical pen and paper method? 

### Background
  T2DM is a chronic metabolic disease involving resistance to the hormone insulin that afflicts nearly 3.8 million in the UK and is one of the most significant comorbidities for a range of diseases including cardiovascular disease, strokes, and obesity. Although it currently has no cure, several factors including weight are known to prevent or even lead to remission of the disease, implying that if we can identify target patients, there is potential to focus efforts on them. It currently costs the NHS billions to treat, meaning there is a great need for the NHS to treat it prophylactically. 
  
  The NHS currently offers a diabetes risk calculator created by diabetes uk, the University of Leicester and University Hospitals of Leicester NHS Trust that allows patients to input their patient parameters such as their ages, genders, ethnicity, etc. This score will then output a percentile risk factor for patients to be diagnosed with diabetes. There are also pen and paper methods that involve simply adding points for each risk factor a patient presents with that are used in a clinical setting. 
  
  There are multiple techniques to apply machine learning to data, including neural networks and random forests. Machine learning is currently being applied to several medical fields and represents outstanding potential to improve patient care and clinical outcomes. The random forest technique from machine learning would be helpful in this study. It is widely used to process the continuous or discrete data to find the best-fit pattern of the data set. ‘Scikit-learn’ is a library that can be used to perform the random forest method.
  
  We can create a 1-layer factor-weighting model that we can solve using pseudoinverse matrices to solve for each factor’s weighting. Here, we will attempt to solve for x where Ax=B, with A being our input parameters and B our risk assessment. Moore-Penrose pseudoinverses allow us to generate the pseudoinverse of matrix A even if it’s not a square matrix (which it won’t be due to our data points). We can find Moore-Penrose pseudoinverses classically with (ATA)-1, but also via Singular Value Decomposition. Here, we found it easier to use Python’s inbuilt functions, but these values could have been relatively easily calculated via Gauss-Jordan Elimination. Fortunately, Python’s numpy library can handle these calculations for us. This is effectively fitting the factors linearly to a curve. 
  
  We will determine whether such methods create a better predictive model than the NHS’s current pencil and paper method, which we will apply to the data via Python. We found patient data collected in China on several hundred T2DM patients that we then used to compare methods. 
  
  While we hope to obtain a more predictive AI model, we also recognize the utility of easy pen and paper models in clinical settings. Thus, we will also try to develop a system requiring only simple math but offers more predictive power than current methods. 

#

This is a test file

