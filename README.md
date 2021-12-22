# Resume_Parsing
Machine Learning Project

**Team**: 
* David Balaban - Team Leader
* Daniel Lichter - Techsmith
* Asma Sadia - Specification Leader
* Maitri Patel - Quality Assurance Specialist

### Business Problem
Keystone Strategy's recruiting team often receives large "resume books"  containing  hundreds of resumes from universities and their student organizations, which they then have to manually parse to catalog attributes about applicants such as education, work experience, skills, etc, before doing a more detailed review for fit with the organization. Keystone would like to automate this process using machine learning. A machine learning-based resume parsing will save recruiting team from hours of daily work by eliminating manual processing and analysis for every resume they receive. 

### Data
Our custom dataset consists of resumes from three resume books from graduate business schools such as Tuck School of Business at Dartmouth, Haas School of Business at Berekely, and Standard Graduate School of Business. In total, there are 841 resumes of MBA candidates.

<img width="650" alt="resume_count_uni" src="https://user-images.githubusercontent.com/20906514/147151896-103f3997-2c89-4830-bf7c-17f05f1b4925.png">


### Conclusion
Selecting the right candidates from a pool of applicants can be one of the toughest jobs for the talent acquisition leaders. Moreover, going through each resume manually for every hiring season can be tiresome and time consuming. The machine learning resume parser tool can be a life saver for the entire company. It can provide unbiased solutions while overcoming possible manual errors.


## How to use our app:

To be able to run our app clone this repository and the first step is to make sure to run the requirements.txt file to install all necessary dependecies:
```
pip install requirements.txt
```
Then it's necessary to install poppler as well to handle the pdfs and there are two ways of doing this:
```
pip install python-poppler
```
or
```
conda install -c conda-forge popper
```
Then once all the necessaries packages have been installed we will run the application using the framework Streamlit by running the following code:
```
python -m streamlit run Resume_Parser.py
```

Here is a screenshot of our application:

<img width="650" alt="app_screenshot" src="https://media.discordapp.net/attachments/890017873324572729/923148182119141406/unknown.png?width=1814&height=982">

