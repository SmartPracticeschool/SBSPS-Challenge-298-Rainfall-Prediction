# SBSPS-Challenge-298-Rainfall-Prediction 
-----------------------------------------
project video link : https://drive.google.com/file/d/1dp-OUkbEnGlPK9GsSkBREnObeBTs114U/view?usp=sharing
 Rainfall Prediction :
-----------------------
The economy of the Indian sub-continent is very much dependent on agriculture.The agricultural yield is dependent on annual and monthly rainfall.Thus rainfall prediction plays a vital role in Indian economy. The farmers can plan their harvest and set up irrigation facility if required. The timely and accurate forecast of rainfall helps to prevent devastating floods. The travel and tourism industry is also very much sensitive to weather conditions. It can properly guide tourists and 	plan travel schedule with the accurate knowledge of rainfall. In this project, we are going to predict Indian Sub Division annual and monthly rainfall based on the previous years rainfall 1901-2017 which is 117 years data. with the help of Machine Learning Algorithm using python and Tensorflow Module, and both are open source.

 Project Scope :
-----------------
The agricultural yield is dependent on annual and monthly rainfall. Thus rainfall	prediction plays a vital role in Indian economy. The farmers can plan their 	harvest and set up irrigation facility if required. The timely and accurate forecast of rainfall helps to prevent devastating floods.The construction of high ways, roads, dams, real estate can be planned cost effectively with the help of rainfall prediction.The travel and tourism industry is also very much sensitive to weather	conditions. It can properly guide tourists and plan travel schedule with the accurate knowledge of rainfall.
	

 Project Planning & Kickoff :
------------------------------
First we are going to study and survey what are the requirement steps and resources and also scope of that project. 
		1. Requirement:
		      i. Python 3.6  Environment
			     ii. Python 3.6  Environment
		2. Python 3.6  Environment:
			    i. Tensorflow 
      ii. Tensorflow_cpu 
      iii. Tensorflow_gpu 
      iv. Scikit-Learn 
      v. Scikit-Image
      vi. Pandas
      vii. Numpy
      vii. Seaborn
      viii. Scipy
      ix. Matplotlib


 Data Collection:
------------------
we are Download dataset from https://data.gov.in/ datatset contain 116 year	Indian Sub Divisional Monthly Rainfall from 1901 to 2017. in this project we using Gangetic West Bengal annual rainfall data to predict annual rainfall.

 Data Preprocessing:
---------------------
Since the dataset contain multiple data but we are only working with Gangetic West Bengal therefore we only taking manually Gangetic West Bengal.
		1. Import The Requirements Libraries.
		2. Import The Dataset :
			 Using Pandas library we are importing the dataset in python.we have csv(comma-separated values) file.
		3. Visualiszation Of The Dataset :
			 In this section we are Visualize  the imported dataset and also see the attributes along with attributes values, and print the values.
		4. Finding Information About The Dataset :
 	   In this section we are finding information about the dataset, using	pandas library. 
  5. Inspect The Dataset :
    	In this section we are Inspect the dataset and plot the data using	seaborn pairplot function.
     
 Split The Dataset Into Training And Testing :
----------------------------------------------
In this section we are Spliting the dataset into training and testing using pandas.
		1. Preprocessing The Dataset:
			   In this section we are using pandas.get_dummies( )function for Preprocess the data before split.
		2. Training Dataset : 
		   	In this section we are taking data for Training using 	pandas.sample()function.
		3. Testing Dataset : 
		   	In this section we are taking data for Testing using pandas.drop() functio.
  4. Labelling The Dataset : 
	     In this section we Labelling The Dataset which is annual attribute
      
 Finding Statistics Value : 
----------------------------
In this section we are finding statistics value which is Mean and Sandard Deviation(STD).
		1 . Statistics Values For Training Dataset :
		     	In this section we are finding Statistics Values for Training Dataset using pandas.describe() function.
		2. Statistics Values For Testing Dataset :
			     In this section we are finding Statistics Values for Testing Dataset using pandas.describe() function.

 Building The Model : 
---------------------
In this section we are Building the model. In this project we are using Sequential model with tensorflow.
		1. Sequential Model : 
			    In this section we are Building the model. In this project we are using Sequential model  with tensorflow, kears API.
		2. Train The Model.
		3. Visualize Model Performance :
			    In this section Visualize  Model Performance and Calculate the Loss, Mean Absolute Error (MAE), Mean Square Error (MSE).
		4. Prediction :
			   In this section after Calculate the Loss, Mean Absolute Error(MAE),	Mean Square Error(MSE),if the model performance good the we are predicting the rainfall.
		5. Plot Actual Vs. Prediction :
			In this section we are plotting Actual vs. Prediction data using matplotlib.

 Application Building :
-----------------------
In this section we are building Application for the project.
		1. Create An HTML File or UI:
			   In this section we are either using IBM cloud platform or we are using local computer.  
		2. Integrate All Python Script :
			   In this section we Integrate All Python Script which is build	previously, With Application.


 Requirement’s:
----------------

• Python 3.7

• Anaconda

• Visual Studio Code

 LINK’S:
---------

• Python : 
----------
Download https://www.python.org/downloads/

• Anaconda : 
------------
Windows:
-------
• Download https://www.anaconda.com/downloads

Linux:
------
Command:
-------
• " wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh "

• " bash Anaconda3-5.3.1-Linux-x86_64.sh "

• " conda update anaconda "

• Visual Studio Code :
----------------------
Download https://code.visualstudio.com/Download

• How to install | Python | | Anaconda | | Opencv library |
------------------------------------------------------------
 [![How to install | Python | | Anaconda | | Opencv library |](https://yt-embed.herokuapp.com/embed?v=eVV3byQlYvA)](https://www.youtube.com/watch?v=eVV3byQlYvA "How to install | Python | | Anaconda | | Opencv library |")


 Installing the required package’s:
-------------------------------------
• pip install -q git+https://github.com/tensorflow/docs 

• conda install -c conda-forge opencv=4.2.0

• pip install scikit-learn

• pip install scikit-image

• pip install matplotlib

• pip install tensorflow

• pip install keras

• pip install pandas

• pip install seaborn

• pip install numpy

• pip install scipy


-----------------------------------------

