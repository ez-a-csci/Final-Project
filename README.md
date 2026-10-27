##Lab 2 Dataset: Optical Recognition of Handwritten Digits
Final Project
Team: Ezra A.

##Type: Classification

##Research Questions 
    Question 1:
    Which digits often get confused with each other, and for what reason?

    Question 2:
    Which pixels/positions across the 8-by-8 grid are most important in discerning the digit? 

    Question 3:
    How does a prediction model's confidence change with objective accuracy, and are inaccurate guesses correlated with stronger-than-average confidence levels?

##Next Steps for Final Project:

I've currently setup base data visualization for averages, as well as set up an early prediction model for identifying digits based on an 8x8 pixel grid as input. Though the code is relatively short, I spent a good amount of time trying to understand how to properly plot my data. Regarding the plots, I had a fun time setting up the data, as well as setting up the visualization to see which digit failed to be predicted by my model. Interestingly, it seems that 8 had the most trouble. This makes sense, since, based on the mean heatmap of pixels, there is a lot of overlap between digits 0, 5, 6, 3, and 9. I would love to try and implement a similar system into a web-based game of some sorts that utilizes its own unique input (not limited specifically to digits but rather abstract symbols). The main issue with this would be battling against compression, since each digit is stored in an 8 by 8 grid and most drawing platforms support variable size canvases.

I would like to experiment with different different prediction models beyond logistic regression and compare output across them this week. I'd like to bring accuracy up (at least for the provided sample) by the end of the week. I'm also interested in learning what the minimum amount of training data the models need to stay accurate (by at least 95%).

Though I am making good progress on this dataset, I would like to explore a larger dataset unrelated to my current selection. Before doing this, however, I would like to discuss with the Professor (preferably both before and after my presentation). If the scope is too large/does not seem achievable, I am happy to stick with this topic. 